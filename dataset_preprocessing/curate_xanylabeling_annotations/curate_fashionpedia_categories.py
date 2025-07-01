from dataset_preprocessing.src.utils import load_json
import os
import json


def curate_fashionpedia_categories_x_any_labeling_files(annotation_directory: str) -> None:
    """
    Curate Fashionpedia categories from the annotation directory and save them to a JSON file.

    Args:
        annotation_directory (str): The directory containing Fashionpedia annotations.
    """
    fashionpedia_supercategory_mapping = {
        "shoe_heels": "shoe", "shoe_ankle": "shoe", "sunglasses": "glasses"}

    annotations_to_delete = ["earring", "necklace", "bracelet", "phone"]

    for file_name in os.listdir(annotation_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(annotation_directory, file_name)
            annotation_dict = load_json(file_path)

            for shape in annotation_dict['shapes']:
                if shape['label'] in fashionpedia_supercategory_mapping:
                    print(
                        f"Mapping {shape['label']} to {fashionpedia_supercategory_mapping[shape['label']]}")
                    shape['label'] = fashionpedia_supercategory_mapping[shape['label']]

            filtered_shapes = [shape for shape in annotation_dict['shapes']
                               if shape['label'] not in annotations_to_delete]

            removed_count = len(
                annotation_dict['shapes']) - len(filtered_shapes)
            if removed_count > 0:
                print(f"Removed {removed_count} annotations from {file_name}")

            annotation_dict['shapes'] = filtered_shapes

            # Write the modified annotation back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_dict, f, indent=4)

# In a COCO annotation file remove category_id == 3,4,5, or 21 AND change all category_id == 1 or 2 to 16 and change category[name] == "sunglasses" to "glasses"


def curate_fashionpedia_categories_coco(annotation_file: str) -> None:

    coco_data = load_json(annotation_file)

    # Step 1: Filter out annotations with category_id 3, 4, 5, or 21
    excluded_ids = {3, 4, 5, 21}
    filtered_annotations = [
        anno for anno in coco_data['annotations']
        if anno['category_id'] not in excluded_ids
    ]

    # Step 2: Change category_id 1 or 2 to 16
    for anno in filtered_annotations:
        if anno['category_id'] in {1, 2}:
            anno['category_id'] = 16

    # Step 3: Change category[name] from "sunglasses" to "glasses"
    for category in coco_data['categories']:
        if category['name'].lower() == "sunglasses":
            category['name'] = "glasses"

    # Step 4: Remove unused categories
    used_category_ids = {anno['category_id'] for anno in filtered_annotations}
    coco_data['categories'] = [
        category for category in coco_data['categories']
        if category['id'] in used_category_ids
    ]

    # Update the annotations
    coco_data['annotations'] = filtered_annotations

    # Overwrite the original file
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Curated annotation file overwritten: {annotation_file}")


if __name__ == "__main__":
    annotation_directory = "dataset_preprocessing/combined_coco_datasets/FashionVeil_supercategories/"
    curate_fashionpedia_categories_x_any_labeling_files(annotation_directory)
    print("Fashionpedia categories curated successfully for X-Any-Labeling annotations.")

    curate_fashionpedia_categories_coco(
        "dataset_preprocessing/combined_coco_datasets/FashionVeil_coco_supercategories.json")
    print("Fashionpedia categories curated successfully for COCO annotations.")
