import os
from glob import glob
from pathlib import Path

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

def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["rtdetr"]
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

def predict_with_onnx(model_name, image_dir, out_dir):
    onnx_path = Path("/home/datsplit/model_development/rtdetr_v2_r101_fashionpedia_b32_split75_25_8.onnx")
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # Load and initialize the feature extractor with specific size
    feature_extractor = RTDetrImageProcessor.from_pretrained(
        "fashionfail/models/configs",
        size={"height": 800, "width": 800},  # Match export dimensions
        # do_resize=True,
        # do_normalize=True,
        # do_rescale=True,
        # do_pad=True,
        max_size=800
    )

    # Run inference on images, accumulate results in a list, save as `.npz` file.
    preds = []
    proba_threshold = 0.5

    logger.debug("Running inference now...")
    for image_path in tqdm(glob(os.path.join(image_dir, "*.jpg"))):
        # Open image
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image with feature extractor
        inputs = feature_extractor(
            images=image,
            return_tensors="pt",
        )
        pixel_values = inputs.pixel_values

        # Run inference
        ort_inputs = {
            'pixel_values': pixel_values.numpy()
        }

        logits, pred_boxes = session.run(['logits', 'pred_boxes'], ort_inputs)

        # Post-process the outputs
        scores = torch.sigmoid(torch.from_numpy(logits))
        # Get predictions above threshold
        max_scores, pred_labels = scores.max(-1)
        mask = max_scores > proba_threshold
        pred_boxes_abs = convert_to_absolute_coordinates(torch.from_numpy(pred_boxes)[mask], image.size)

        filtered_boxes = extended_box_convert(
            pred_boxes_abs, in_fmt="cxcywh", out_fmt="xyxy"
        )
        filtered_scores = max_scores[mask]
        filtered_labels = pred_labels[mask]

        # Accumulate results.
        preds.append(
            {
                "image_file": Path(image_path).name,
                "boxes": filtered_boxes.numpy(),
                "classes": filtered_labels.numpy(),
                "scores": filtered_scores.numpy(),
            }
        )
    # Save results in a compressed `.npz` file
    os.makedirs(out_dir, exist_ok=True)
    out_file_name = model_name + ".npz"
    np.savez_compressed(os.path.join(out_dir, out_file_name), data=preds)
    logger.debug(f"Results are saved at: {out_dir + out_file_name}")

if __name__ == "__main__":
    # Parse args
    parser = get_cli_args_parser()
    args = parser.parse_args()

    # call the respective function
    predict_with_onnx(args.model_name, args.image_dir, args.out_dir)
