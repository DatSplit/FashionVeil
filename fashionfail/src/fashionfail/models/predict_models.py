import os
from glob import glob
from pathlib import Path

import json
import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from pycocotools import mask as mask_api
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

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
        choices=["facere_base", "facere_plus"],
        help="Name of the model to run inference, either `facere_base` or `facere_plus`.",
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
        help="If set, will map the IDs to match FashionVeil category IDs.",
    )

    return parser


def predict_with_onnx(model_name: str, image_dir: str, out_dir: str, fashionveil_mapping: bool) -> None:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    path_to_onnx = hf_hub_download(
        repo_id="rizavelioglu/fashionfail",
        filename=f"{model_name}.onnx",
        repo_type="model",
    )
    ort_session = onnxruntime.InferenceSession(
        path_to_onnx, providers=[
            "CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    preds = []
    proba_threshold = 0.5

    logger.debug("Running inference now...")
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
    for image in tqdm(image_files):
        
        img = read_image(image)
        
        if img.shape[0] == 4:
            img = img[:3, ...]
        elif img.shape[0] == 1:
            img = img.repeat(3, 0)
        img_transformed = transforms(img)
       
        ort_inputs = {
            ort_session.get_inputs()[0].name: img_transformed.unsqueeze(
                dim=0).numpy()
        }
        ort_outs = ort_session.run(None, ort_inputs)
        boxes, labels, scores, masks = ort_outs

        masks = masks.squeeze(1)
        filtered_masks = masks > proba_threshold
        filtered_labels = labels  
        filtered_boxes = boxes
        filtered_scores = scores
        filtered_encoded_masks = [
            mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
            for mask in filtered_masks
        ]

        if fashionveil_mapping:
            with open("/home/datsplit/model_development/fashionveil_coco.json", "r") as f:
                new_mapping = json.load(f)
            new_mapping = new_mapping["categories"]
            new_mapping_dict = {item["id"]: item["name"]
                                for item in new_mapping}
            reverse_new_mapping_dict = {
                v: k for k, v in new_mapping_dict.items()}
            mapped_labels = []
            valid_indices = []

            for idx, id in enumerate(filtered_labels.tolist()):
                original_name = ORIGINAL_CLASSES_MAPPING_DICT[id-1]
                if original_name in new_mapping_dict.values():
                    mapped_id = reverse_new_mapping_dict[original_name]
                    mapped_labels.append(mapped_id)
                    valid_indices.append(idx)
            
            filtered_labels = np.array(torch.tensor(mapped_labels))
            filtered_boxes = boxes[valid_indices]
            filtered_scores = scores[valid_indices]
            filtered_encoded_masks = [
                filtered_encoded_masks[i] for i in valid_indices]

        
        boxes_tensor = torch.tensor(filtered_boxes)
        
        boxes_tensor = extended_box_convert(
            boxes_tensor, in_fmt="xyxy", out_fmt="yxyx")
        
        preds.append(
            {
                "image_file": Path(image).name,
                "boxes": boxes_tensor.numpy(),
                "classes": filtered_labels,
                "scores": filtered_scores,
                "masks": filtered_encoded_masks,
            }
        )

    
    os.makedirs(out_dir, exist_ok=True)
    out_file_name = model_name + ".npz"
    np.savez_compressed(os.path.join(out_dir, out_file_name), data=preds)
    logger.debug(
        f"Results are saved at: {os.path.join(out_dir, out_file_name)}")


if __name__ == "__main__":
    parser = get_cli_args_parser()
    args = parser.parse_args()

    predict_with_onnx(args.model_name, args.image_dir,
                      args.out_dir, args.fashionveil_mapping)
