import cv2
import torch
import time
import threading
from transformers import YolosForObjectDetection, YolosFeatureExtractor
from PIL import Image
import streamlit as st

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

# Load the model and feature extractor
MODEL_NAME = "DatSplit/yolos-base-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
model_fashion = YolosForObjectDetection.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model_fashion = model_fashion.to('cuda')

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open video stream.")
    st.stop()
st.logo("Middel+18.png")
# Streamlit GUI Elements
st.title("Self-service divest (security lane)")
detected_label = st.empty()

frame_placeholder = st.empty()

# Detection tracker
detected_items_tracker = {}
last_detection_time = 0


def detect_items(frame):
    """Detect items using the model."""
    inputs = feature_extractor(images=frame, return_tensors="pt").to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model_fashion(**inputs)
    results = feature_extractor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([frame.shape[:2]])
    )[0]
    detected_items = {
        categories[label.item()].lower()
        for label, score in zip(results['labels'], results['scores'])
        if score > 0.4 and categories[label.item()].lower() in items_to_check
    }
    print('det', detected_items)
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
    # detected_label.text(f"Please remove your {', '.join(items_to_display)}." if items_to_display else "No items detected.")
    # items_text.color("red" if items_to_display else "green")
    sorted_items = sorted(
        items_to_display, key=lambda x: items_to_check.index(x))
    if sorted_items:
        items_text = []
        for item in sorted_items:
            if item in ["bag, wallet"]:
                items_text.append(f"<span style='font-size:30px;'>{item_emoticons.get(
                    item, '')} Put your bag(s) on top of your other items</span>")
            else:
                items_text.append(
                    f"<span style='font-size:30px;'>{item_emoticons.get(item, '')} {item}</span>")

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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - last_detection_time >= 0.1:  # Process every 0.2 seconds
            detected_items = detect_items(frame)
            st.session_state['detected_items'] = detected_items
            last_detection_time = time.time()
            update_gui(st.session_state['detected_items'])
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pil_image = Image.fromarray(frame_rgb)
        # frame_placeholder.image(pil_image)


# Initialize session state
if 'detected_items' not in st.session_state:
    st.session_state['detected_items'] = set()

# Run video stream in a separate thread
# threading.Thread(target=video_stream, daemon=True).start()

# Run video stream in the main thread
video_stream()

# Update GUI with detected items

# img_file_buffer = st.camera_input("")
