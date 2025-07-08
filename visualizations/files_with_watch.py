import json
import re
import os
import shutil
from pathlib import Path


def find_files_with_category_id_direct(annotation_file_path, target_category_id=13):
    """
    Find all file names that contain annotations with the specified category_id
    by directly parsing the JSON file.
    """
    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)

    target_image_ids = set()
    for annotation in coco_data.get('annotations', []):
        if annotation.get('category_id') == target_category_id:
            target_image_ids.add(annotation['image_id'])

    file_names = []
    for image in coco_data.get('images', []):
        if image['id'] in target_image_ids:
            file_names.append(image['file_name'])

    def extract_number(filename):
        match = re.search(r'p(\d+)', filename)
        return int(match.group(1)) if match else 0

    return sorted(file_names, key=extract_number)


files_with_category_13 = find_files_with_category_id_direct(
    "/home/datsplit/FashionVeil/FashionVeil_all/FashionVeil_supercategories.json", 13
)
print(f"Files with category_id 13: {files_with_category_13}")
distinct_p_numbers = set()
for filename in files_with_category_13:
    match = re.search(r'p(\d+)', filename)
    if match:
        distinct_p_numbers.add(int(match.group(1)))

print(
    f"Number of distinct p numbers with category_id 13: {len(distinct_p_numbers)}")
print(f"Total individual files: {len(files_with_category_13)}")

smart_watches = [
    "p1", "p4", "p6", "p7", "p9", "p10", "p11", "p14", "p15", "p23", "p25", "p26", "p28", "p30",
    "p33", "p37", "p39", "p42", "p43", "p44", "p45", "p53", "p58", "p59", "p60", "p65", "p66", "p67",
    "p68", "p70", "p73", "p75", "p76", "p87", "p89", "p95", "p98", "p109", "p121", "p123", "p125",
    "p126", "p127", "p138", "p144"
]
print(len(smart_watches)/len(distinct_p_numbers))
# Filter images that start with smart_watches patterns
smart_watch_images = []
for filename in files_with_category_13:
    # Extract the p number from the filename
    match = re.search(r'^(p\d+)_', filename)
    if match:
        p_number = match.group(1)
        if p_number in smart_watches:
            smart_watch_images.append(filename)

print(f"Smart watch images found: {len(smart_watch_images)}")
print(f"Smart watch images: {smart_watch_images}")

# Create output directory
output_dir = "smart_watch_images"
os.makedirs(output_dir, exist_ok=True)

# Source directory containing the images
source_dir = "/home/datsplit/FashionVeil/FashionVeil_all"

# Copy smart watch images to the new folder
copied_count = 0
for filename in smart_watch_images:
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(output_dir, filename)

    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        # print(f"Copied: {filename}")
    else:
        print(f"File not found: {source_path}")

print(f"\nTotal images copied: {copied_count}")
print(f"Images saved to: {os.path.abspath(output_dir)}")
