import json
import argparse
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Filter predictions to only those with image_ids in the test set.")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to COCO-format predictions JSON")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to test set COCO JSON")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to save filtered predictions JSON")
    args = parser.parse_args()

    # 1. Load valid image_ids from test set
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    test_image_ids = set(img["id"] for img in test_data["images"])

    # 2. Load predictions (COCO format: list of dicts)
    with open(args.pred_path, "r") as f:
        preds = json.load(f)

    # 3. Filter predictions by image_id
    filtered_preds = [
        pred for pred in preds if pred["image_id"] in test_image_ids]

    # 4. Save filtered predictions as JSON
    with open(args.out_path, "w") as f:
        json.dump(filtered_preds, f)

    logger.info(
        f"Filtered predictions saved to {args.out_path} ({len(filtered_preds)} entries)")


if __name__ == "__main__":
    main()
