import json


def replace_category_ids_with_direct_mapping(annotation_file_path, new_categories, output_file_path):
    """
    Replace category IDs using a direct category list.

    Args:
        annotation_file_path: Path to the annotation file to be modified
        new_categories: List of new category dictionaries
        output_file_path: Path where the modified annotation file will be saved
    """

    with open(annotation_file_path, 'r') as f:
        annotation_data = json.load(f)

    name_to_new_id = {}
    for category in new_categories:
        name_to_new_id[category['name']] = category['id']

    old_id_to_new_id = {}
    for old_category in annotation_data['categories']:
        old_name = old_category['name']
        old_id = old_category['id']

        if old_name in name_to_new_id:
            new_id = name_to_new_id[old_name]
            old_id_to_new_id[old_id] = new_id
            print(f"Mapping: '{old_name}' ID {old_id} -> {new_id}")
        else:
            print(
                f"Warning: Category '{old_name}' not found in new categories")
            old_id_to_new_id[old_id] = old_id

    annotation_data['categories'] = new_categories

    updated_annotations = 0
    for annotation in annotation_data['annotations']:
        old_category_id = annotation['category_id']
        if old_category_id in old_id_to_new_id:
            annotation['category_id'] = old_id_to_new_id[old_category_id]
            updated_annotations += 1

    print(f"Updated {updated_annotations} annotations")

    with open(output_file_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)

    print(f"Updated annotation file saved to: {output_file_path}")


if __name__ == "__main__":

    new_categories = [
        {
            "id": 1,
            "name": "bag, wallet",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "belt",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "cardigan",
            "supercategory": ""
        },
        {
            "id": 4,
            "name": "coat",
            "supercategory": ""
        },
        {
            "id": 5,
            "name": "glasses",
            "supercategory": ""
        },
        {
            "id": 6,
            "name": "hat",
            "supercategory": ""
        },
        {
            "id": 7,
            "name": "hood",
            "supercategory": ""
        },
        {
            "id": 8,
            "name": "jacket",
            "supercategory": ""
        },
        {
            "id": 9,
            "name": "scarf",
            "supercategory": ""
        },
        {
            "id": 10,
            "name": "shoe",
            "supercategory": ""
        },
        {
            "id": 11,
            "name": "top, t-shirt, sweatshirt",
            "supercategory": ""
        },
        {
            "id": 12,
            "name": "vest",
            "supercategory": ""
        },
        {
            "id": 13,
            "name": "watch",
            "supercategory": ""
        },
        {
            "id": 14,
            "name": "zipper",
            "supercategory": ""
        }
    ]

    replace_category_ids_with_direct_mapping(
        'FashionVeil_test/fashionveil_test.json',
        new_categories,
        'fashionveil_test.json'
    )
