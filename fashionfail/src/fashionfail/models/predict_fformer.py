import sys
import os
from glob import glob
from pathlib import Path
import json
import mmcv
import numpy as np
import torch
from loguru import logger
from mmdet.apis import inference_detector, init_detector
from pycocotools import mask as mask_api
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(
    0, "/home/datsplit/model_development/fashionfail/FashionFormer")


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
        "--model_path",
        type=str,
        required=True,
        help="The path to the checkpoint/trained model.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The path to model configs.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset, which will be included in the output filename.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="The output directory where the predictions file will be saved.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="The image directory for prediction.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum score of bboxes to be shown.",
    )

    parser.add_argument(
        "--fashionveil_mapping",
        type=bool,
        default=False,
        help="If set, will map the IDs to match FashionVeil category IDs.",
    )
    return parser


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros(
        (n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        if x.shape[0] == 0:
            continue

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def predict(
    model_path: str,
    config_path: str,
    dataset_name: str,
    out_dir: str,
    image_dir: str,
    score_threshold: float,
    device,
    fashionveil_mapping: bool,
) -> None:
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, model_path, device=device)

    # Get image paths from `image_dir` (support all common image types)
    img_exts = ["*.jpg", "*.jpeg", "*.png"]
    img_list = []
    for ext in img_exts:
        img_list.extend(glob(os.path.join(image_dir, ext)))

    # Accumulate results in a list, save as `.npz` file.
    preds = []
    logger.debug("Running inference now...")
    for image in tqdm(img_list):
        # Run inference for a single image
        result = inference_detector(model, [image])

        # Parse result --> list[tuple(bbox, masks, attributes)]
        bbox_result, segm_result, _ = result[0]

        # Process labels
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # Process segmentation masks
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

        # Process scores
        scores = np.vstack(bbox_result)[:, -1]

        # Filter results based on threshold
        if score_threshold:
            inds = scores > score_threshold
            scores = scores[inds]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        # Get boxes from masks
        boxes = masks_to_boxes(torch.from_numpy(segms))

        # Process masks
        encoded_masks = [
            mask_api.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in segms
        ]
        if fashionveil_mapping:
            home_dir = Path.home()
            fashionveil_coco_path = home_dir / "FashionVeil" / "fashionveil_coco.json"
            with open(fashionveil_coco_path, "r") as f:
                new_mapping = json.load(f)
            new_mapping = new_mapping["categories"]
            new_mapping_dict = {item["id"]: item["name"]
                                for item in new_mapping}
            reverse_new_mapping_dict = {
                v: k for k, v in new_mapping_dict.items()}
            # Map and filter valid indices
            mapped_labels = []
            valid_indices = []

            for idx, id in enumerate(labels.tolist()):
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
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = torch.tensor(mapped_labels)
            # logger.info(f"{labels}, og {labels}")
        # Accumulate results.
        preds.append(
            {
                "image_file": Path(image).name,
                "boxes": boxes.numpy(),
                "classes": labels,
                "scores": scores,
                "masks": encoded_masks,
            }
        )

    # Save results in a compressed `.npz` file: 'model_name-dataset_name.npz'
    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists
    out_file_name = f"{Path(model_path).stem}-{dataset_name}.npz"
    out_file_path = os.path.join(out_dir, out_file_name)
    np.savez_compressed(out_file_path, data=preds)
    logger.debug(f"Results are saved at: {out_file_path}")


if __name__ == "__main__":
    # Parse args
    args = get_cli_args_parser().parse_args()
    logger.debug(f"Parsed args: {args}")
    # setting device on GPU if available
    # Pass as string for MMDetection compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Run inference and store results
    predict(
        args.model_path,
        args.config_path,
        args.dataset_name,
        args.out_dir,
        args.image_dir,
        args.score_threshold,
        device,
        args.fashionveil_mapping
    )
