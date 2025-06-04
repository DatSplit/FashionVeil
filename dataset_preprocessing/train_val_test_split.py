import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from loguru import logger
import json
from dataset_preprocessing.src.utils import load_json
import shutil

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
ORIGINAL_CATEGORIES = [
    {"id": k, "name": v, "supercategory": ""} for k, v in ORIGINAL_CLASSES_MAPPING_DICT.items()
]
NAME_TO_ORIGINAL_ID = {v: k for k, v in ORIGINAL_CLASSES_MAPPING_DICT.items()}


def groupwise_split(images, group_key='file_name', train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    # Extract group labels (e.g., passenger IDs)
    groups = [img[group_key].split("_")[0]
              for img in images]  # Use the whole file_name as group
    unique_groups = np.unique(groups)
    print(f"Unique groups found: {unique_groups}, total {len(unique_groups)}")
    # Split groups
    train_groups, temp_groups = train_test_split(
        unique_groups, train_size=train_size, random_state=random_state)
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=test_size/(test_size+val_size), random_state=random_state)
    logger.info(
        f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}, Test groups: {len(test_groups)}")

    def filter_images(groups_set):
        return [img for img, grp in zip(images, groups) if grp in groups_set]

    train_images = filter_images(set(train_groups))
    val_images = filter_images(set(val_groups))
    test_images = filter_images(set(test_groups))

    return train_images, val_images, test_images


def train_val_test_split_export(annotation_file, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, prepare_for_rfdetr='rfdetr'):
    """
    Splits COCO dataset into train, validation, and test sets using groupwise split.

    Parameters:
    - annotation_file: str, path to COCO annotation file
    - train_size, val_size, test_size: float, split ratios (must sum to 1.0)
    - random_state: int, random seed

    Returns:
    - None (writes split files to disk)
    """
    coco_data = load_json(annotation_file)
    images = coco_data['images']

    train_images, val_images, test_images = groupwise_split(
        images, group_key='file_name', train_size=train_size, val_size=val_size, test_size=test_size, random_state=random_state
    )
    # Credits to FashionFail.

    def filter_and_export_data(data, split_images, target_file):
        img_ids = {img['id'] for img in split_images}
        anns = pd.DataFrame(data["annotations"])
        imgs = pd.DataFrame(data["images"])
        anns_filtered = anns[anns.image_id.isin(img_ids)]
        imgs_filtered = imgs[imgs.id.isin(img_ids)]
        data["annotations"] = anns_filtered.to_dict("records")
        data["images"] = imgs_filtered.to_dict("records")

        # --- Remap categories and annotation category_ids if rfdetr ---
        if prepare_for_rfdetr == 'rfdetr':
            # Replace categories
            data["categories"] = ORIGINAL_CATEGORIES

            # Build old id -> name mapping from original categories
            old_id_to_name = {cat["id"]: cat["name"]
                              for cat in coco_data["categories"]}
            # Remap annotation category_ids
            for ann in data["annotations"]:
                old_name = old_id_to_name.get(ann["category_id"])
                if old_name in NAME_TO_ORIGINAL_ID:
                    ann["category_id"] = NAME_TO_ORIGINAL_ID[old_name]
                else:
                    ann["category_id"] = -1  # or skip
            # Remove annotations with unknown category
            data["annotations"] = [
                ann for ann in data["annotations"] if ann["category_id"] != -1]

        with open(target_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, separators=(",", ":"), indent=4)

    if prepare_for_rfdetr == 'rfdetr':
        logger.info("Preparing dataset for rfdetr fine-tuning.")
        target_dir = "./FashionVeil/"
        os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "valid"), exist_ok=True)
        filter_and_export_data(coco_data.copy(), train_images,
                               os.path.join(target_dir + "train/", "_annotations.coco.json"))
        filter_and_export_data(coco_data.copy(), val_images,
                               os.path.join(target_dir + "valid/", "_annotations.coco.json"))
        filter_and_export_data(coco_data.copy(), test_images,
                               os.path.join(target_dir, "fashionveil_test.json"))
        image_src_dir = target_dir
        for img in train_images:
            src = os.path.join(image_src_dir, img['file_name'])
            dst = os.path.join(target_dir, "train", img['file_name'])
            if os.path.exists(src):
                shutil.move(src, dst)
        for img in val_images:
            src = os.path.join(image_src_dir, img['file_name'])
            dst = os.path.join(target_dir, "valid", img['file_name'])
            if os.path.exists(src):
                shutil.move(src, dst)
    else:
        target_dir = os.path.dirname(annotation_file)
        filter_and_export_data(coco_data.copy(), train_images,
                               os.path.join(target_dir, "fashionveil_train.json"))
        filter_and_export_data(coco_data.copy(), val_images,
                               os.path.join(target_dir, "fashionveil_val.json"))
        filter_and_export_data(coco_data.copy(), test_images,
                               os.path.join(target_dir, "fashionveil_test.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split COCO dataset into train, validation, and test sets.")
    parser.add_argument("annotation_file", type=str,
                        help="Path to the COCO annotation file (.json)")
    parser.add_argument("--train_size", type=float, default=0.6,
                        help="Proportion for training set (default: 0.6)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion for validation set (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion for test set (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--prepare_for_rfdetr", type=str, default=None,
                        help="prepare for rfdetr, if set to 'rfdetr' will prepare the dataset for rfdetr fine-tuning")

    args = parser.parse_args()

    train_val_test_split_export(
        args.annotation_file, args.train_size, args.val_size, args.test_size, args.random_state, args.prepare_for_rfdetr
    )
# python dataset_preprocessing/train_val_test_split.py dataset_preprocessing/annotations/fashionveil_coco.json --train_size 0.6 --val_size 0.2 --test_size 0.2
