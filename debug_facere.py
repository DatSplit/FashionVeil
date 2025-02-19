from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from huggingface_hub import hf_hub_download
import onnxruntime
from PIL import Image
import json
import random


import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torchvision.transforms.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageColor
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def fix_category_id(cat_ids: list):
    # Define the excluded category ids and the remaining ones
    excluded_indices = {2, 12, 16, 19, 20}
    remaining_categories = list(set(range(27)) - excluded_indices)

    # Create a dictionary that maps new IDs to old(original) IDs
    new_id_to_org_id = dict(
        zip(range(len(remaining_categories)), remaining_categories))

    return [new_id_to_org_id[i-1]+1 for i in cat_ids]


def process_categories() -> tuple:
    """
    Load and process category information from a JSON file.
    Returns a tuple containing two dictionaries: `category_id_to_name` maps category IDs to their names, and
    `category_id_to_color` maps category IDs to a randomly sampled RGB color.
    Returns:
        tuple: A tuple containing two dictionaries:
            - `category_id_to_name`: a dictionary mapping category IDs to their names.
            - `category_id_to_color`: a dictionary mapping category IDs to a randomly sampled RGB color.
    """
    # Load raw categories from JSON file
    with open("categories.json") as fp:
        categories = json.load(fp)

    # Map category IDs to names
    category_id_to_name = {d["id"]: d["name"] for d in categories}

    # Set the seed for the random sampling operation
    random.seed(42)

    # Get a list of all the color names in the PIL colormap
    color_names = list(ImageColor.colormap.keys())

    # Sample 46 unique colors from the list of color names
    sampled_colors = random.sample(color_names, 46)

    # Convert the color names to RGB values
    rgb_colors = [ImageColor.getrgb(color_name)
                  for color_name in sampled_colors]

    # Map category IDs to colors
    category_id_to_color = {
        category["id"]: color for category, color in zip(categories, rgb_colors)
    }

    return category_id_to_name, category_id_to_color


model_name = "facere+"
path_onnx = hf_hub_download(
    repo_id="rizavelioglu/fashionfail",
    filename="facere_plus.onnx",  # or "facere_plus.onnx"
)
categories = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat',
    'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt',
    'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette',
    'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle',
    'sequin', 'tassel'
]

# Items to check for
items_to_check = ['coat', 'cape', 'sweater', 'cardigan', 'jacket', 'vest',
                  'hood', 'scarf', 'hat', 'bag, wallet', 'belt', 'watch', 'zipper']
# Load pre-trained model transformations.
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

# Load image and apply original transformation to the image.
img = Image.open("t.png").convert("RGB")
img_transformed = transforms(img)


# Create an inference session.
ort_session = onnxruntime.InferenceSession(
    path_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Run inference on the input.
ort_inputs = {
    ort_session.get_inputs()[0].name: img_transformed.unsqueeze(dim=0).numpy()
}
ort_outs = ort_session.run(None, ort_inputs)

# Parse the model output.
boxes, labels, scores, masks = ort_outs
# Map label IDs to names and colors
label_id_to_name, label_id_to_color = process_categories()

# Filter out predictions using thresholds
labels_id = labels[scores > 0.5].tolist()
if model_name == "facere+":
    labels_id = fix_category_id(labels_id)
# models output is in range: [1,class_id+1], hence re-map to: [0,class_id]
labels = [label_id_to_name[int(i) - 1] for i in labels_id]
masks = (masks[scores > 0.5] > 0.5).astype(np.uint8)
boxes = boxes[scores > 0.5]

print('det', labels, scores)
