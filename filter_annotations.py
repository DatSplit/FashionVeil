import json
import argparse
from typing import Dict, List, Set


def filter_coco_annotations(input_file: str, output_file: str):
    """
    Filter COCO annotations to keep only specified categories and merge shoe classes.
    """
    # Define the categories to keep
    categories_to_keep = {
        'top', 't-shirt', 'sweatshirt', 'cardigan', 'jacket', 'vest', 'coat',
        'hat', 'watch', 'belt', 'scarf', 'hood', 'earring', 'necklace',
        'bracelet', 'shoe_heels', 'shoe_ankle', 'sunglasses', 'bag', 'wallet'
    }

    # Load the COCO annotation file
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # Create mapping for category names to IDs that we want to keep
    old_to_new_category_id = {}
    new_categories = []
    new_category_id = 0

    # Process categories
    for category in coco_data['categories']:
        cat_name = category['name']

        if cat_name in categories_to_keep:
            if cat_name in ['shoe_heels', 'shoe_ankle']:
                # Merge both shoe types into 'shoe_forbidden'
                if 'shoe_forbidden' not in [c['name'] for c in new_categories]:
                    new_categories.append({
                        'id': new_category_id,
                        'name': 'shoe_forbidden',
                        'supercategory': category.get('supercategory', '')
                    })
                    shoe_forbidden_id = new_category_id
                    new_category_id += 1
                else:
                    # Find existing shoe_forbidden ID
                    shoe_forbidden_id = next(
                        c['id'] for c in new_categories if c['name'] == 'shoe_forbidden')

                old_to_new_category_id[category['id']] = shoe_forbidden_id
            else:
                # Keep other categories as they are
                new_categories.append({
                    'id': new_category_id,
                    'name': cat_name,
                    'supercategory': category.get('supercategory', '')
                })
                old_to_new_category_id[category['id']] = new_category_id
                new_category_id += 1

    # Get the set of valid category IDs
    valid_category_ids = set(old_to_new_category_id.keys())

    # Filter annotations
    filtered_annotations = []
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in valid_category_ids:
            # Update category ID to new mapping
            annotation['category_id'] = old_to_new_category_id[annotation['category_id']]
            filtered_annotations.append(annotation)

    # Get image IDs that have valid annotations
    valid_image_ids = set(ann['image_id'] for ann in filtered_annotations)

    # Filter images to keep only those with valid annotations
    filtered_images = [img for img in coco_data['images']
                       if img['id'] in valid_image_ids]

    # Create the filtered COCO data
    filtered_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': new_categories,
        'images': filtered_images,
        'annotations': filtered_annotations
    }

    # Save the filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_coco_data, f, indent=2)

    # Print statistics
    print(f"Original categories: {len(coco_data['categories'])}")
    print(f"Filtered categories: {len(new_categories)}")
    print(f"Original annotations: {len(coco_data['annotations'])}")
    print(f"Filtered annotations: {len(filtered_annotations)}")
    print(f"Original images: {len(coco_data['images'])}")
    print(f"Filtered images: {len(filtered_images)}")

    print("\nKept categories:")
    for cat in new_categories:
        print(f"  - {cat['name']} (ID: {cat['id']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter COCO annotations file')
    parser.add_argument(
        'input_file', help='Path to input COCO annotation file')
    parser.add_argument(
        'output_file', help='Path to output filtered COCO annotation file')

    args = parser.parse_args()

    filter_coco_annotations(args.input_file, args.output_file)
