import rfdetr.datasets.transforms as T
import os
from glob import glob
from pathlib import Path
import json

import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from torchvision.io import read_image
from tqdm import tqdm
from transformers import RTDetrImageProcessor
from PIL import Image

from fashionfail.utils import extended_box_convert
ORIGINAL_CLASSES_MAPPING_DICT = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
    26: "umbrella",
    27: "hood",
    28: "collar",
    29: "lapel",
    30: "epaulette",
    31: "sleeve",
    32: "pocket",
    33: "neckline",
    34: "buckle",
    35: "zipper",
    36: "applique",
    37: "bead",
    38: "bow",
    39: "flower",
    40: "fringe",
    41: "ribbon",
    42: "rivet",
    43: "ruffle",
    44: "sequin",
    45: "tassel"
}


def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["rfdetr"]
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        default=None,
        help="The image directory for prediction.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="The directory where predictions will be saved.",
    )
    parser.add_argument(
        "--fashionveil_mapping",
        type=bool,
        default=False,
        help="If set, will map the IDs to match FashionVeil category IDs."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for filtering predictions."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./rfdetrl_best.onnx",
        help="Path to the ONNX model file."
    )
    return parser


def convert_to_absolute_coordinates(pred_boxes, image_size):
    """
    Convert relative coordinates to absolute pixel coordinates.

    Args:
        pred_boxes: Tensor with predicted boxes in relative coordinates (cx, cy, w, h).
        image_size: Tuple (width, height) of the image.

    Returns:
        Tensor with absolute coordinates (cx, cy, w, h).
    """
    width, height = image_size
    pred_boxes_abs = pred_boxes.clone()

    # Convert from relative [0-1] to absolute pixel coordinates
    # For center coordinates (cx, cy)
    pred_boxes_abs[:, 0] *= width   # cx (center x)
    pred_boxes_abs[:, 1] *= height  # cy (center y)

    # For dimensions (w, h)
    pred_boxes_abs[:, 2] *= width   # width
    pred_boxes_abs[:, 3] *= height  # height

    return pred_boxes_abs


def predict_with_onnx(model_name, image_dir, out_dir, fashionveil_mapping, confidence_threshold, onnx_path):
    onnx_path = Path(onnx_path) #"/home/datsplit/model_development/rfdetrl_best.onnx"
    session = onnxruntime.InferenceSession(str(onnx_path),
                                           providers=[
                                               'CUDAExecutionProvider', 'CPUExecutionProvider']
                                           )

    # Define preprocessing transforms
    transforms = T.Compose([
        T.SquareResize([1120]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preds = []
    proba_threshold = confidence_threshold
    logger.debug(type(proba_threshold))
    logger.debug(f"Running inference now... with threshold: {proba_threshold}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob(os.path.join(image_dir, ext)))
    all_image_files = sorted(list(set(all_image_files)))

    for image_path in tqdm(all_image_files):
        image = Image.open(image_path).convert("RGB")
        tensor_img, _ = transforms(image, None)
        tensor_img = tensor_img.unsqueeze(0)  # Add batch dim: [1, 3, H, W]

        ort_inputs = {
            'input': tensor_img.cpu().numpy()
        }

        pred_boxes, logits = session.run(['dets', 'labels'], ort_inputs)
        # print(logits, pred_boxes)

        # Convert logits to torch tensor and apply sigmoid activation
        scores = torch.sigmoid(torch.from_numpy(logits))
        max_scores, pred_labels = scores.max(-1)
        mask = max_scores > proba_threshold

        pred_boxes = torch.from_numpy(pred_boxes[0])  # shape: [300, 4]
        image_w, image_h = image.size

        # Convert to absolute pixel coordinates
        pred_boxes_abs = pred_boxes.clone()
        pred_boxes_abs[:, 0] *= image_w  # cx
        pred_boxes_abs[:, 1] *= image_h  # cy
        pred_boxes_abs[:, 2] *= image_w  # w
        pred_boxes_abs[:, 3] *= image_h  # h

        # Fix mask shape
        mask = mask.squeeze(0)  # shape becomes [300]

        # Apply mask and convert to xyxy format
        filtered_boxes = extended_box_convert(
            pred_boxes_abs[mask], in_fmt="cxcywh", out_fmt="xyxy")
        filtered_scores = max_scores.squeeze(0)[mask]
        filtered_labels = pred_labels.squeeze(0)[mask]
        # filtered_labels.numpy()
        if fashionveil_mapping:
            # logger.info("Mapping labels to FashionVeil category IDs...")
            # logger.info("Mapping labels to FashionVeil category IDs...")
            # Map labels to FashionVeil category_ids
            # open .json with new category ids
            with open("/home/datsplit/model_development/fashionveil_coco.json", "r") as f:
                new_mapping = json.load(f)
            new_mapping = new_mapping["categories"]
            new_mapping_dict = {item["id"]: item["name"]
                                for item in new_mapping}
            reverse_new_mapping_dict = {
                v: k for k, v in new_mapping_dict.items()}
            # Map and filter valid indices
            mapped_labels = []
            valid_indices = []

            for idx, id in enumerate(filtered_labels.tolist()):
                original_name = ORIGINAL_CLASSES_MAPPING_DICT[id]
                if original_name in new_mapping_dict.values():
                    mapped_id = reverse_new_mapping_dict[original_name]
                    mapped_labels.append(mapped_id)
                    valid_indices.append(idx)
                else:
                    continue
                    # logger.info(
                    #     f"Dropping ID {id} as {original_name} not in FashionVeil categories")

            # Apply filtered indices
            filtered_boxes = filtered_boxes[valid_indices]
            filtered_scores = filtered_scores[valid_indices]
            filtered_labels = torch.tensor(mapped_labels)

        # logger.info(Path(image_path).name)
        preds.append({
            "image_file": Path(image_path).name,
            "boxes": filtered_boxes.numpy(),
            "classes": filtered_labels.numpy(),
            "scores": filtered_scores.numpy(),
        })

    os.makedirs(out_dir, exist_ok=True)
    if confidence_threshold == 0.5:
        out_file_name = model_name + ".npz"
    else:
        out_file_name = model_name + "_" + str(confidence_threshold) + ".npz"
    out_path = os.path.join(out_dir, out_file_name)
    np.savez_compressed(out_path, data=preds)
    logger.debug(f"Results are saved at: {out_path}")


if __name__ == "__main__":
    # Parse args
    parser = get_cli_args_parser()
    args = parser.parse_args()

    # call the respective function
    print(args.fashionveil_mapping)
    predict_with_onnx(args.model_name, args.image_dir,
                      args.out_dir, args.fashionveil_mapping, args.confidence_threshold, args.onnx_path)
