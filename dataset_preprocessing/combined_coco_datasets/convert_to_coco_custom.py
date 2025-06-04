"""
This script is adapted from X-AnyLabeling project.
Original code: https://github.com/CVHub520/X-AnyLabeling/blob/c57412141e177d8e58e00c9360c53f80290e80f4/tools/label_converter.py

Licensed under GNU General Public License v3.0 (GPL-3.0)
"""

import os
import json
import os.path as osp
from datetime import date
from tqdm import tqdm
from loguru import logger
VERSION = "1.0"

MAPPING = {
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

print("Script started")


class AnnotationConverter:
    def __init__(self, classes_file, output_file_name="instances_default.json"):
        logger.debug(
            f"(AnnotationConverter init called)")
        if classes_file:
            with open(classes_file, "r", encoding="utf-8") as f:
                self.classes = f.read().splitlines()
                logger.info(
                    f"Loaded classes from {classes_file}: {self.classes}")
        else:
            self.classes = []
        print(f"import classes is: {self.classes}")

        self.output_file_name = output_file_name

    def map_occlusion_level_to_category(self, occlusion_level: float) -> str:
        if occlusion_level <= 0.25:
            return 'No to slight occlusion'
        elif occlusion_level <= 0.5:
            return 'Moderate occlusion'
        elif occlusion_level <= 0.75:
            return 'Heavy occlusion'
        elif occlusion_level <= 1.0:
            return 'Extreme occlusion'
        else:
            raise ValueError(
                f"Invalid occlusion level: {occlusion_level}. Must be between 0 and 1.")

    def get_coco_data(self):
        coco_data = {
            "info": {
                "year": 2023,
                "version": VERSION,
                "description": "COCO Label Conversion",
                "contributor": "CVHub",
                "url": "https://github.com/CVHub520/X-AnyLabeling",
                "date_created": str(date.today()),
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://www.gnu.org/licenses/gpl-3.0.html",
                    "name": "GNU GENERAL PUBLIC LICENSE Version 3",
                }
            ],
            "categories": [],
            "images": [],
            "annotations": [],
        }
        return coco_data

    def custom_to_coco(self, input_path, output_path):
        coco_data = self.get_coco_data()
        logger.info(self.classes)
        for i, class_name in enumerate(self.classes):
            coco_data["categories"].append(
                {"id": i + 1, "name": class_name, "supercategory": ""}
            )

        image_id = 0
        annotation_id = 0

        file_list = os.listdir(input_path)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            image_id += 1

            input_file = osp.join(input_path, file_name)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_path = data["imagePath"]

            # image_name = osp.splitext(osp.basename(image_path))[0] + ".jpg"

            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": image_path,
                    "width": data["imageWidth"],
                    "height": data["imageHeight"],
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            for shape in data["shapes"]:
                annotation_id += 1
                label = shape["label"]
                points = shape["points"]
                difficult = shape.get("difficult", False)
                class_id = self.classes.index(label)
                x_min = min(points[0][0], points[2][0])
                y_min = min(points[0][1], points[2][1])
                x_max = max(points[0][0], points[2][0])
                y_max = max(points[0][1], points[2][1])
                width = x_max - x_min
                height = y_max - y_min
                if shape["description"] is not None and shape["description"] != "":
                    occlusion_level = self.map_occlusion_level_to_category(
                        float(shape["description"]))
                else:
                    occlusion_level = "No to slight occlusion"
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "ignore": int(difficult),
                    "segmentation": [],
                    "occlusion": occlusion_level
                }

                coco_data["annotations"].append(annotation)

        output_file = osp.join(output_path, self.output_file_name)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def extract_classes(self, input_path):
        """Extract unique class names from all JSON files"""
        classes = set()
        file_list = os.listdir(input_path)

        for file_name in tqdm(
            file_list, desc="Extracting classes", unit="file", colour="blue"
        ):
            if not file_name.endswith(".json"):
                continue

            input_file = osp.join(input_path, file_name)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for shape in data["shapes"]:
                classes.add(shape["label"])

        return sorted(list(classes))

    def convert(self, input_path, output_path):
        """Main conversion function"""
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Extract and set classes
        self.classes = self.extract_classes(input_path)
        print(f"Found {len(self.classes)} classes: {', '.join(self.classes)}")

        # Convert to COCO format
        self.custom_to_coco(input_path, output_path)
        print(
            f"Conversion complete. COCO file saved at: {osp.join(output_path, self.output_file_name)}")


if __name__ == "__main__":
    input_dir = "/home/datsplit/wearables_detection_airport_security/dataset_preprocessing/combined_coco_datasets/FashionVeil_supercategories"
    output_dir = "/home/datsplit/wearables_detection_airport_security/dataset_preprocessing/annotations/"
    os.makedirs(output_dir, exist_ok=True)

    converter = AnnotationConverter(
        "/home/datsplit/wearables_detection_airport_security/fashionpedia_classes.txt", "fashionveil_coco.json")
    converter.convert(input_dir, output_dir)
