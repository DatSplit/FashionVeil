import json


def map_occlusion_level_to_category(occlusion_level: float) -> str:
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


def map_label_to_category_id(annotations: dict, label_name: str) -> int:
    """
    Maps the label name to a category ID from the annotations file.

    Args:
        annotations_path (str): Path to the annotations file.
        label_name (str): The name of the label to find (e.g., "shoe_heels").

    Returns:
        int: Category ID corresponding to the label name.

    Raises:
        ValueError: If the label_name is not found in the categories.
    """

    for category in annotations['categories']:
        if category['name'] == label_name:
            return category['id']

    raise ValueError(
        f"Label '{label_name}' not found in categories")
