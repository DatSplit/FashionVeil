import json
import os
import glob
from pathlib import Path


def cleanup_unused_images(json_file_path, image_directory=None):
    """
    Delete PNG images that are in the directory but not referenced in the COCO JSON file.

    Args:
        json_file_path (str): Path to the .coco.json file
        image_directory (str, optional): Directory containing images. If None, uses same directory as JSON file.
    """

    # Use same directory as JSON file if image_directory not specified
    if image_directory is None:
        image_directory = os.path.dirname(json_file_path)

    # Load the COCO JSON file
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    # Extract all file names from the JSON
    referenced_files = set()
    for image in coco_data.get('images', []):
        file_name = image.get('file_name', '')
        if file_name:
            referenced_files.add(file_name)

    print(f"Found {len(referenced_files)} images referenced in {json_file_path}")

    # Get all PNG files in the directory
    png_pattern = os.path.join(image_directory, "*.png")
    all_png_files = glob.glob(png_pattern)
    all_png_filenames = {os.path.basename(f) for f in all_png_files}

    print(f"Found {len(all_png_files)} PNG files in {image_directory}")

    # Find files to delete (in directory but not in JSON)
    files_to_delete = all_png_filenames - referenced_files

    if not files_to_delete:
        print("✅ No unused PNG files found. All images are referenced in the JSON file.")
        return

    print(f"\n🗑️  Found {len(files_to_delete)} unused PNG files:")
    for file_name in sorted(files_to_delete):
        print(f"  - {file_name}")

    # Ask for confirmation
    response = input(
        f"\nDo you want to delete these {len(files_to_delete)} files? (y/N): ")

    if response.lower() in ['y', 'yes']:
        deleted_count = 0
        for file_name in files_to_delete:
            file_path = os.path.join(image_directory, file_name)
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"✅ Deleted: {file_name}")
            except FileNotFoundError:
                print(f"⚠️  File not found: {file_name}")
            except Exception as e:
                print(f"❌ Error deleting {file_name}: {e}")

        print(f"\n🎉 Successfully deleted {deleted_count} unused PNG files!")
    else:
        print("Operation cancelled.")


# Usage example
if __name__ == "__main__":
    json_file = "/home/datsplit/FashionVeil/FashionVeil_test_supercategories/fashionveil_test.json"
    image_dir = "/home/datsplit/FashionVeil/FashionVeil_test_supercategories"

    cleanup_unused_images(json_file, image_dir)
