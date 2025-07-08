import numpy as np
import json
import argparse
import os
from pathlib import Path

ORIGINAL_CLASSES_MAPPING_DICT = {
    1: "shirt, blouse",
    2: "top, t-shirt, sweatshirt",
    3: "sweater",
    4: "cardigan",
    5: "jacket",
    6: "vest",
    7: "pants",
    8: "shorts",
    9: "skirt",
    10: "coat",
    11: "dress",
    12: "jumpsuit",
    13: "cape",
    14: "glasses",
    15: "hat",
    16: "headband, head covering, hair accessory",
    17: "tie",
    18: "glove",
    19: "watch",
    20: "belt",
    21: "leg warmer",
    22: "tights, stockings",
    23: "sock",
    24: "shoe",
    25: "bag, wallet",
    26: "scarf",
    27: "umbrella",
    28: "hood",
    29: "collar",
    30: "lapel",
    31: "epaulette",
    32: "sleeve",
    33: "pocket",
    34: "neckline",
    35: "buckle",
    36: "zipper",
    37: "applique",
    38: "bead",
    39: "bow",
    40: "flower",
    41: "fringe",
    42: "ribbon",
    43: "rivet",
    44: "ruffle",
    45: "sequin",
    46: "tassel"
}


def main():
    parser = argparse.ArgumentParser(
        description="Map and filter AMRCNN prediction labels to FashionVeil class labels.")
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to input .npy or .npz predictions file')
    parser.add_argument('--output_file', type=str,
                        required=True, help='Path to output .npy file')
    args = parser.parse_args()

    home_dir = Path.home()
    fashionveil_coco_path = home_dir / "FashionVeil" / "fashionveil_coco.json"
    with open(fashionveil_coco_path, "r") as f:
        new_mapping = json.load(f)
    new_mapping = new_mapping["categories"]
    new_mapping_dict = {item["id"]: item["name"] for item in new_mapping}
    reverse_new_mapping_dict = {v: k for k, v in new_mapping_dict.items()}

    data = np.load(args.predictions_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        results = data["data"]
    else:
        results = data

    for item in results:

        if "image_file" in item:
            item["image_file"] = os.path.basename(item["image_file"])
        filtered_boxes = np.array(item["boxes"])
        filtered_scores = np.array(item["scores"])
        filtered_labels = np.array(item["classes"])

        filtered_attributes = np.array(item.get("attributes", []))
        filtered_masks = item.get("masks", [])

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

        item["boxes"] = filtered_boxes[valid_indices]
        item["scores"] = filtered_scores[valid_indices]
        item["classes"] = np.array(mapped_labels)
        if filtered_attributes.size > 0:
            item["attributes"] = filtered_attributes[valid_indices]
        if filtered_masks:
            item["masks"] = [filtered_masks[i] for i in valid_indices]

    np.save(args.output_file, results)


if __name__ == "__main__":
    main()
