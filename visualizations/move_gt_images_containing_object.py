import os
import shutil
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO dataset by classes and move images.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the images directory.")
    parser.add_argument("--annotations_file", type=str,
                        required=True, help="Path to COCO annotations JSON file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where filtered images will be moved.")
    parser.add_argument(
        "--target_classes",
        type=str,
        nargs="+",
        required=True,
        help="List of target class names (e.g. --target_classes bag wallet)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.annotations_file, "r") as f:
        coco = json.load(f)

    id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    selected_image_ids = set()
    for ann in coco["annotations"]:
        cat_name = id_to_name[ann["category_id"]]
        if cat_name in args.target_classes:
            selected_image_ids.add(ann["image_id"])

    for image_id in selected_image_ids:
        filename = id_to_filename[image_id]
        src = os.path.join(args.images_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)

    print(f"Moved {len(selected_image_ids)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
