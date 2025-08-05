import json
import os


def fix_coco_indexing(annotation_file):
    """
    Convert COCO format annotations from 1-based to 0-based category indexing
    """
    print(f"Processing: {annotation_file}")

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Get original category info
    original_cats = sorted(data['categories'], key=lambda x: x['id'])
    print(f"Original category IDs: {[cat['id'] for cat in original_cats]}")
    print(
        f"Original range: {min(cat['id'] for cat in original_cats)} to {max(cat['id'] for cat in original_cats)}")

    # Create mapping from old IDs (1-based) to new IDs (0-based)
    id_mapping = {}
    for i, cat in enumerate(original_cats):
        old_id = cat['id']
        new_id = i  # 0-based indexing
        id_mapping[old_id] = new_id
        cat['id'] = new_id
        print(f"Category '{cat['name']}': {old_id} -> {new_id}")

    # Update all annotation category_ids
    print(f"\nUpdating {len(data['annotations'])} annotations...")
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in id_mapping:
            ann['category_id'] = id_mapping[old_cat_id]
        else:
            print(
                f"Warning: Found annotation with unknown category_id: {old_cat_id}")

    # Verify the changes
    new_cat_ids = set(ann['category_id'] for ann in data['annotations'])
    print(f"New category IDs in annotations: {sorted(new_cat_ids)}")
    print(f"New range: 0 to {max(new_cat_ids)}")
    print(f"Total categories: {len(data['categories'])}")

    # Save the fixed file
    output_file = annotation_file.replace('.json', '_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Fixed file saved as: {output_file}")
    return output_file


def fix_all_annotation_files(dataset_dir):
    """
    Fix all annotation files in the dataset directory
    """
    annotation_files = [
        '_annotations.coco.json',
    ]

    fixed_files = []
    for ann_file in annotation_files:
        file_path = os.path.join(dataset_dir, ann_file)
        if os.path.exists(file_path):
            fixed_file = fix_coco_indexing(file_path)
            fixed_files.append(fixed_file)
        else:
            print(f"File not found: {file_path}")

    return fixed_files


if __name__ == "__main__":
    dataset_dir = "/home/datsplit/.cache/rfdetr_fashionpedia-divest/valid/"

    # Fix all annotation files
    fixed_files = fix_all_annotation_files(dataset_dir)

    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for fixed_file in fixed_files:
        print(f"✓ Fixed: {fixed_file}")

    print("\nNow update your training script to use the *_fixed.json files!")
    print("Also verify that num_classes matches the number of categories (should be the same).")
