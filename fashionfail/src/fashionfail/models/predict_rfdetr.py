import rfdetr.datasets.transforms as T
import os
from glob import glob
from pathlib import Path
import json

import numpy as np
import onnxruntime
import torch
from loguru import logger
from tqdm import tqdm
from PIL import Image

from fashionfail.utils import extended_box_convert, ORIGINAL_CLASSES_MAPPING_DICT, FASHIONPEDIA_DIVEST_CLASSES_MAPPING_DICT


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
        help="The image directory for inferencing.",
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

    parser.add_argument(
        "--fashionpedia_divest_mapping",
        type=bool,
        default=False,
        help="If set, will use the Fashionpedia divest mapping."
    )
    return parser


def predict_with_onnx(model_name, image_dir, out_dir, fashionveil_mapping, confidence_threshold, onnx_path, fashionpedia_divest_mapping):
    onnx_path = Path(onnx_path)
    session = onnxruntime.InferenceSession(str(onnx_path),
                                           providers=[
                                               'CUDAExecutionProvider', 'CPUExecutionProvider']
                                           )

    transforms = T.Compose([
        T.SquareResize([1120]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    preds = []
    proba_threshold = confidence_threshold
    logger.debug(f"Running inference now...")

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob(os.path.join(image_dir, ext)))
    all_image_files = sorted(list(set(all_image_files)))

    for image_path in tqdm(all_image_files):
        image = Image.open(image_path).convert("RGB")
        tensor_img, _ = transforms(image, None)
        tensor_img = tensor_img.unsqueeze(0)

        ort_inputs = {
            'input': tensor_img.cpu().numpy()
        }

        pred_boxes, logits = session.run(['dets', 'labels'], ort_inputs)

        scores = torch.sigmoid(torch.from_numpy(logits))
        max_scores, pred_labels = scores.max(-1)
        mask = max_scores > proba_threshold

        pred_boxes = torch.from_numpy(pred_boxes[0])
        image_w, image_h = image.size

        pred_boxes_abs = pred_boxes.clone()
        pred_boxes_abs[:, 0] *= image_w
        pred_boxes_abs[:, 1] *= image_h
        pred_boxes_abs[:, 2] *= image_w
        pred_boxes_abs[:, 3] *= image_h

        mask = mask.squeeze(0)

        filtered_boxes = extended_box_convert(
            pred_boxes_abs[mask], in_fmt="cxcywh", out_fmt="xyxy")
        filtered_scores = max_scores.squeeze(0)[mask]
        filtered_labels = pred_labels.squeeze(0)[mask]

        if fashionpedia_divest_mapping:
            home_dir = Path.home()
            conversion_path = home_dir / "FashionVeil" / "fashionpedia_divest_mapping.json"
        if fashionveil_mapping:
            home_dir = Path.home()
            conversion_path = home_dir / "FashionVeil" / "fashionveil_coco.json"

        if fashionveil_mapping or fashionpedia_divest_mapping:
            with open(conversion_path, "r") as f:
                new_mapping = json.load(f)
            new_mapping = new_mapping["categories"]
            new_mapping_dict = {item["id"]: item["name"]
                                for item in new_mapping}
            reverse_new_mapping_dict = {
                v: k for k, v in new_mapping_dict.items()}

            mapped_labels = []
            valid_indices = []

            for idx, id in enumerate(filtered_labels.tolist()):

                if fashionveil_mapping and not fashionpedia_divest_mapping:
                    original_name = ORIGINAL_CLASSES_MAPPING_DICT[id]
                if fashionpedia_divest_mapping:
                    original_name = FASHIONPEDIA_DIVEST_CLASSES_MAPPING_DICT[id]

                if original_name in new_mapping_dict.values():
                    mapped_id = reverse_new_mapping_dict[original_name]
                    mapped_labels.append(mapped_id)
                    valid_indices.append(idx)
                else:
                    continue

            filtered_boxes = filtered_boxes[valid_indices]
            filtered_scores = filtered_scores[valid_indices]
            filtered_labels = torch.tensor(mapped_labels)

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
    parser = get_cli_args_parser()
    args = parser.parse_args()

    print(args.fashionveil_mapping)
    predict_with_onnx(args.model_name, args.image_dir,
                      args.out_dir, args.fashionveil_mapping, args.confidence_threshold, args.onnx_path, args.fashionpedia_divest_mapping)
