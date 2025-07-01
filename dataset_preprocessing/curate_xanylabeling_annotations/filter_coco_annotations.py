import json
import os
from pathlib import Path


def filter_coco_annotations(source_json_path, reference_json_path, output_json_path):
    """
    Filter COCO annotations from source_json to only include annotations 
    for images that are present in the reference_json file.
    Preserves ALL annotation fields including 'occlusion'.

    Args:
        source_json_path (str): Path to the COCO JSON file to filter (contains full annotations with occlusion)
        reference_json_path (str): Path to the reference COCO JSON file (determines which images to keep)
        output_json_path (str): Path where the filtered JSON will be saved
    """

    # Load the reference JSON to get available images
    print(f"Loading reference file: {reference_json_path}")
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)

    # Extract image filenames from reference
    reference_images = set()
    for image in reference_data.get('images', []):
        file_name = image.get('file_name', '')
        if file_name:
            reference_images.add(file_name)

    print(f"Found {len(reference_images)} images in reference file")

    # Load the source JSON to filter
    print(f"Loading source file: {source_json_path}")
    with open(source_json_path, 'r') as f:
        source_data = json.load(f)

    # Filter images
    filtered_images = []
    filtered_image_ids = set()
    image_id_mapping = {}  # old_id -> new_id

    for image in source_data.get('images', []):
        file_name = image.get('file_name', '')
        if file_name in reference_images:
            new_id = len(filtered_images) + 1
            old_id = image['id']
            image_id_mapping[old_id] = new_id

            # Create a copy of the image and update its ID
            filtered_image = image.copy()
            filtered_image['id'] = new_id
            filtered_images.append(filtered_image)
            filtered_image_ids.add(old_id)

    print(f"Filtered to {len(filtered_images)} images that exist in reference")

    # Filter annotations - PRESERVE ALL FIELDS including 'occlusion'
    filtered_annotations = []
    for annotation in source_data.get('annotations', []):
        image_id = annotation.get('image_id')
        if image_id in filtered_image_ids:
            # Create a complete copy of the annotation to preserve all fields
            filtered_annotation = annotation.copy()

            # Update only the IDs
            filtered_annotation['id'] = len(filtered_annotations) + 1
            filtered_annotation['image_id'] = image_id_mapping[image_id]

            # All other fields (including 'occlusion', 'bbox', 'area', etc.) are preserved
            filtered_annotations.append(filtered_annotation)

    print(f"Filtered to {len(filtered_annotations)} annotations")

    # Check if occlusion field is present in annotations
    occlusion_count = sum(
        1 for ann in filtered_annotations if 'occlusion' in ann)
    print(f"Annotations with occlusion field: {occlusion_count}")

    # Create filtered dataset
    filtered_data = {
        'info': source_data.get('info', {}),
        'licenses': source_data.get('licenses', []),
        'categories': source_data.get('categories', []),
        'images': filtered_images,
        'annotations': filtered_annotations
    }

    # Save filtered data
    print(f"Saving filtered data to: {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print("✅ Filtering complete!")
    print(f"Summary:")
    print(f"  - Original images: {len(source_data.get('images', []))}")
    print(
        f"  - Original annotations: {len(source_data.get('annotations', []))}")
    print(f"  - Filtered images: {len(filtered_images)}")
    print(f"  - Filtered annotations: {len(filtered_annotations)}")

    # Show sample annotation to verify fields are preserved
    if filtered_annotations:
        print(f"\nSample filtered annotation:")
        sample = filtered_annotations[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    source_json = "/home/datsplit/FashionVeil/dataset_preprocessing/annotations/fashionveil_all_coco.json"
    reference_json = "/home/datsplit/FashionVeil/FashionVeil_test/fashionveil_test.json"
    output_json = "/home/datsplit/FashionVeil/FashionVeil_test_all.json"

    filter_coco_annotations(source_json, reference_json, output_json)
