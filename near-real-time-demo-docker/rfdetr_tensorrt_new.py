import os
import time
import json
import cv2

import torch

import numpy as np
import streamlit as st
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda

import rfdetr.datasets.transforms as T

st.set_page_config(layout="wide")

items_to_check = [
    'coat', 'cape', 'sweater', 'cardigan', 'jacket', 'vest',
    'hood', 'scarf', 'hat', 'bag, wallet', 'belt', 'watch', 'zipper'
]

os.environ["CUDA_CACHE_DISABLE"] = "0"
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483647"
os.environ["CUDA_CACHE_PATH"] = "./cuda_cache"


def process_categories() -> dict:
    """
    Process the categories from the JSON file and create a mapping from
    category IDs to category names.
    Returns:
        dict: A dictionary mapping category IDs to category names.
    """

    with open("categories.json", "r", encoding="utf-8") as fp:
        categories = json.load(fp)
    category_id_to_name = {d["id"]: d["name"] for d in categories}

    return category_id_to_name


label_id_to_name = process_categories()

# Preprocessing
transforms = T.Compose([
    T.SquareResize([1120]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TensorRT engine setup
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_file_path: str) -> trt.ICudaEngine | None:
    """
    Load a serialized engine from file.

    Args:
        engine_file_path (str): Path to the serialized engine file.

    Returns:
        trt.ICudaEngine | None: A TensorRT ICudaEngine object or None if loading failed.

    Examples:
        >>> engine = load_engine("your_onnx_model.engine")
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


@st.cache_resource
def load_trt_engine_context() -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """
    Load the TensorRT engine and create an execution context.

    Returns:
        tuple[trt.ICudaEngine, trt.IExecutionContext]: A tuple containing the TensorRT engine and execution context.
    """

    print("Loading TensorRT engine...")
    engine = load_engine("rfdetrl_best.engine")
    context = engine.create_execution_context()
    print("TensorRT engine loaded.")
    return engine, context


engine, context = load_trt_engine_context()

# Synchronous inference function using execute_v2


def run_sync(context: trt.IExecutionContext, engine: trt.ICudaEngine, input_data: np.ndarray) -> dict:
    """
    Perform synchronous inference using TensorRT's execute_v2 method.

    Args:
        context (trt.IExecutionContext): The TensorRT execution context.
        engine (trt.ICudaEngine): The TensorRT engine.
        input_data (np.ndarray): Input data in NumPy array format.

    Returns:
        dict: A dictionary containing the output tensor names and their corresponding data.
    """
    # Allocate memory for input and output tensors
    bindings = {}
    bindings_addr = {}

    # Process input tensors
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]

    for name in input_names:
        tensor_shape = context.get_tensor_shape(name)
        tensor_size = trt.volume(tensor_shape) * np.dtype(np.float32).itemsize
        device_input = cuda.mem_alloc(tensor_size)
        cuda.memcpy_htod(
            device_input, np.ascontiguousarray(input_data).ravel())
        bindings[name] = device_input
        bindings_addr[name] = int(device_input)

    # Process output tensors
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    for name in output_names:
        tensor_shape = context.get_tensor_shape(name)
        tensor_size = trt.volume(tensor_shape) * np.dtype(np.float32).itemsize
        device_output = cuda.mem_alloc(tensor_size)
        bindings[name] = device_output
        bindings_addr[name] = int(device_output)

    # Run inference synchronously using execute_v2
    context.execute_v2(list(bindings_addr.values()))

    # Retrieve outputs
    outputs = {}
    for name in output_names:
        tensor_shape = context.get_tensor_shape(name)
        output_host = cuda.pagelocked_empty(
            trt.volume(tensor_shape), dtype=np.float32)
        cuda.memcpy_dtoh(output_host, bindings[name])
        outputs[name] = output_host.reshape(tensor_shape)

    return outputs

# Detection function using run_sync for inference


def detect_items(frame):
    frame = Image.fromarray(frame)
    img_transformed, _ = transforms(frame, None)
    img_transformed = img_transformed.unsqueeze(0)
    input_np = img_transformed.numpy().astype(np.float32)

    output_tensors = run_sync(context, engine, input_np)
    pred_boxes = output_tensors[list(output_tensors.keys())[
        0]].reshape((1, -1, 4))
    logits = output_tensors[list(output_tensors.keys())[1]].reshape(
        (1, -1, len(label_id_to_name)))

    scores = torch.sigmoid(torch.from_numpy(logits))
    max_scores, pred_labels = scores.max(-1)

    confidence_mask = max_scores.squeeze(0) > 0.7
    filtered_labels = pred_labels.squeeze(0)[confidence_mask]
    labels_id = filtered_labels.tolist()

    labels = [label_id_to_name[int(i)] for i in labels_id]
    detected_items = {
        label.lower()
        for label in labels
        if label.lower() in items_to_check
    }

    return detected_items

# Streamlit UI setup


st.title("Deposit the following items in the tray in front of you:")
detected_label = st.empty()
frame_placeholder = st.empty()
item_emoticons = {
    'coat': 'item_images/coat.png',
    'cape': 'item_images/cape.png',
    'sweater': 'item_images/sweater.png',
    'cardigan': 'item_images/cardigan.png',
    'jacket': 'item_images/jacket.png',
    'vest': 'item_images/vest.png',
    'hood': 'item_images/hood.png',
    'scarf': 'item_images/scarf.png',
    'hat': 'item_images/hat.png',
    'bag, wallet': 'item_images/bag.png',
    'belt': 'item_images/belt.png',
    'watch': 'item_images/watch.png',
    'zipper': 'item_images/zipper.png'
}
detected_items_tracker = {}
last_detection_time = 0


def update_gui(detected_items_current_frame):  # Renamed arg for clarity
    current_time = time.time()

    # Update tracker for items detected in the current frame
    for item in detected_items_current_frame:
        detected_items_tracker[item] = current_time

    # List of {'name': item, 'style': style_str, 'is_active_for_combo': bool}
    processed_items_for_display = []

    items_to_purge_from_tracker = []

    tracker_items_snapshot = list(detected_items_tracker.items())

    for item, last_seen_time in tracker_items_snapshot:
        time_since_last_seen = current_time - last_seen_time
        style = ""  # Default style (black text)
        is_active_for_combo = False

        if item in detected_items_current_frame:  # Item is actively being detected now
            style = ""
            is_active_for_combo = True
        # Not in current frame, but seen recently (still "active" display)
        elif time_since_last_seen <= 10:
            style = ""
            is_active_for_combo = True  # Still considered active for combo logic
        # Not in current frame, seen >10s ago but <=20s ago (deposited)
        elif time_since_last_seen <= 20:
            style = "color:green;"
            # is_active_for_combo remains False for green items
        else:  # Item is too old, mark for purging and skip display
            items_to_purge_from_tracker.append(item)
            continue

        processed_items_for_display.append(
            {'name': item, 'style': style, 'is_active_for_combo': is_active_for_combo})

    # Purge very old items from tracker
    for item_to_purge in items_to_purge_from_tracker:
        # Check if item still exists and its last_seen_time qualifies for purging
        if item_to_purge in detected_items_tracker and \
           (current_time - detected_items_tracker[item_to_purge]) > 20:
            del detected_items_tracker[item_to_purge]

    # Sort items based on the predefined order in items_to_check
    sorted_display_entries = sorted(
        processed_items_for_display,
        key=lambda x: items_to_check.index(
            x['name']) if x['name'] in items_to_check else float('inf')
    )

    items_html_parts = []

    # --- Combo logic (only for active items) ---
    active_combo_item_names = [
        entry['name'] for entry in sorted_display_entries if entry['is_active_for_combo']]

    if "bag, wallet" in active_combo_item_names and "belt" in active_combo_item_names:
        items_html_parts.append(
            "<div style='display: flex; align-items: center; margin-bottom: 10px;'><img src='item_images/belt.png' width='50' height='50' style='margin-right: 10px;'><span style='font-size:30px;'>Please put your belt in your bag</span></div>")

    if "bag, wallet" in active_combo_item_names and "watch" in active_combo_item_names:
        items_html_parts.append(
            "<div style='display: flex; align-items: center; margin-bottom: 10px;'><img src='item_images/watch.png' width='50' height='50' style='margin-right: 10px;'><span style='font-size:30px;'>Please put your watch in your bag</span></div>")

    if "bag, wallet" in active_combo_item_names and "hat" in active_combo_item_names:
        items_html_parts.append(
            "<div style='display: flex; align-items: center; margin-bottom: 10px;'><img src='item_images/hat.png' width='50' height='50' style='margin-right: 10px;'><span style='font-size:30px;'>Please put your hat in your bag</span></div>")
    # --- End Combo logic ---

    # --- Individual item display ---
    for entry in sorted_display_entries:
        item_name = entry['name']
        style_str = entry['style']
        image_path = item_emoticons.get(item_name, '')

        display_text = item_name  # Default display text
        # Special messages for certain items
        if item_name == "bag, wallet":
            display_text = "Put your bag(s) on top of your other items"
        # Removed duplicate "jacket"
        elif item_name in ["jacket", "coat", "cape", "sweater", "cardigan", "vest"]:
            display_text = f"Put your {item_name} at the bottom of the tray"

        if image_path:
            items_html_parts.append(
                f"<div style='display: flex; align-items: center; margin-bottom: 10px;'><img src='{image_path}' width='50' height='50' style='margin-right: 10px;'><span style='font-size:30px; {style_str}'>{display_text}</span></div>")
        else:  # Fallback if no image path, though unlikely with current setup
            items_html_parts.append(
                f"<div style='display: flex; align-items: center; margin-bottom: 10px;'><span style='font-size:30px; {style_str}'>{display_text}</span></div>")

    if items_html_parts:
        detected_label.markdown("<br>".join(
            items_html_parts), unsafe_allow_html=True)
    else:
        detected_label.markdown(
            "<span style='color:green; font-size:50px;'>No items to deposit or all items processed.</span>",
            unsafe_allow_html=True
        )


# Start video
device_path = "dev/video4"
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap or not cap.isOpened():
    st.error("Error: Could not open video stream.")
    st.stop()


def video_stream():
    global last_detection_time
    skip_frames = 0
    start_time = time.time()
    frames_inferenced = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_frames > 0:
            skip_frames -= 1
            continue

        if time.time() - last_detection_time >= 0.0001:
            detected_items = detect_items(frame)
            frames_inferenced += 1
            print(f"FPS: {frames_inferenced / (time.time() - start_time)}")
            st.session_state['detected_items'] = detected_items
            last_detection_time = time.time()
            skip_frames = 5
            update_gui(detected_items)


# Session state init
if 'detected_items' not in st.session_state:
    st.session_state['detected_items'] = set()

video_stream()
