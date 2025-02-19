import cv2
import torch
import time
import threading
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from huggingface_hub import hf_hub_download
import onnxruntime
from PIL import Image
import streamlit as st
import tempfile
from torchvision.transforms import ToTensor
import json
import random
from PIL import Image, ImageColor
import numpy as np
from torchvision.utils import draw_bounding_boxes
import uuid
import torchvision.transforms.functional as F
import random


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


# List of categories from the dataset
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

# Load the ONNX model
path_onnx = hf_hub_download(
    repo_id="rizavelioglu/fashionfail",
    filename="facere_base.onnx",  # or "facere_plus.onnx"
)
model_name = "facerebase"
# Load pre-trained model transformations.
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
label_id_to_name, label_id_to_color = process_categories()
# Create an inference session.
ort_session = onnxruntime.InferenceSession(
    path_onnx, providers=["CUDAExecutionProvider"] #, "CPUExecutionProvider"
)
fps = 0
import os
def draw_predictions(labels, scores, img, boxes):
    # Convert image tensor to uint8
    img = (img * 255).byte()
    img = torch.from_numpy(img.numpy())
    random_image_name = f"{uuid.uuid4()}.png"
    # Draw bounding boxes
    img_bbox = draw_bounding_boxes(
        img, boxes=torch.from_numpy(boxes), labels=labels, width=4)
    imgs_list = []
    img_bbox = F.to_pil_image(img_bbox)
    save_path = os.path.join("/app/saved_images", random_image_name)
    img_bbox.save(save_path)
    imgs_list.append("img_bbox.png")
    print("Image saved to:", save_path)
    return img_bbox


# Start video capture
# cap = cv2.VideoCapture(0)
cap = None
for device in [0, "/dev/video4"]:
    try:
        temp_cap = cv2.VideoCapture(device)
        if temp_cap.isOpened():
            cap = temp_cap
    except:
        pass
if not cap.isOpened():
    st.error("Error: Could not open video stream.")
    st.stop()
#st.logo("Middel+18.png")
# Streamlit GUI Elements
st.title("Self-service divest (security lane)")

detected_label = st.empty()

frame_placeholder = st.empty()

# Detection tracker
detected_items_tracker = {}
last_detection_time = 0


def detect_items(frame):
    if isinstance(frame, str):
        img = read_image(frame)
    else:
        img = ToTensor()(frame)
        # Save the image array to a temporary file
        # with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        #     cv2.imwrite(temp_file.name, frame)
        #     img = read_image(temp_file.name)

    img_transformed = transforms(img)
    ort_inputs = {
        ort_session.get_inputs()[0].name: img_transformed.unsqueeze(
            dim=0).numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    boxes, labels, scores, masks = ort_outs

    labels_id = labels[scores > 0.95].tolist()
    if model_name == "facere+":
        labels_id = fix_category_id(labels_id)

    # models output is in range: [1,class_id+1], hence re-map to: [0,class_id]
    labels = [label_id_to_name[int(i) - 1] for i in labels_id]
    # masks = (masks[scores > 0.5] > 0.5).astype(np.uint8)
    boxes = boxes[scores > 0.95]

    # Create a set of detected items based on the filtered labels
    detected_items = {label.lower()
                      for label in labels if label.lower() in items_to_check}

    print('Detected items:', detected_items)

    if random.random() < 0.20 and len(detected_items) > 0:
        # Filter the boxes, scores, and labels to only include the detected items
        print("saving image")
        filtered_indices = [i for i, label in enumerate(
            labels) if label.lower() in detected_items]
        filtered_labels = [labels[i] for i in filtered_indices]
        filtered_boxes = boxes[filtered_indices]
        filtered_scores = scores[filtered_indices]

        draw_predictions(filtered_labels, filtered_scores, img, filtered_boxes)
    return detected_items


item_emoticons = {
    'coat': '🧥',
    'cape': '🦸',
    'sweater': '🥼',
    'cardigan': '🧥',
    'jacket': '👔',
    'vest': '🦺',
    'hood': '🧑🏿‍🚒',
    'scarf': '🧣',
    'hat': '🎩',
    'bag, wallet': '👜',
    'belt': '➰',
    'watch': '⌚',
    'zipper': '🔗'
}


def update_gui(detected_items):
    """Update the GUI with detected items."""
    current_time = time.time()
    for item in detected_items:
        detected_items_tracker[item] = current_time
    items_to_display = [
        item for item, last_time in detected_items_tracker.items()
        if current_time - last_time <= 10
    ]
    print(items_to_display)
    sorted_items = sorted(
        items_to_display, key=lambda x: items_to_check.index(x))

    if sorted_items:
        items_text = []
        for item in sorted_items:
            if item in ["bag, wallet"]:
                items_text.append(f"<span style='font-size:30px;'>{item_emoticons.get(item, '')} Put your bag(s) on top of your other items</span>")
            else:
                items_text.append(f"<span style='font-size:30px;'>{item_emoticons.get(item, '')} {item}</span>")

        detected_label.markdown(
            "<span style='font-size:25px;'>Please remove the following items and put them in the tray in front of you:</span><br>" +
            "<br>".join(items_text),
            unsafe_allow_html=True
        )
    else:
        detected_label.markdown(
            "<span style='color:green; font-size:30px;'>No items detected</span>",
            unsafe_allow_html=True
        )


def video_stream():
    """Capture frames and update the GUI."""
    global last_detection_time
    start_time = time.time()
    frames_inferenced = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - last_detection_time >= 0.0001:
            detected_items = detect_items(frame)
            frames_inferenced += 1
            print(f"FPS: {frames_inferenced / (time.time() - start_time)}")
            st.session_state['detected_items'] = detected_items
            last_detection_time = time.time()
            update_gui(detected_items)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # pil_image = Image.fromarray(frame_rgb)
            # frame_placeholder.image(pil_image)


# Initialize session state
if 'detected_items' not in st.session_state:
    st.session_state['detected_items'] = set()

video_stream()
