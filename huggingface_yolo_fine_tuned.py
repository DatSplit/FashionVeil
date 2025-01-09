import cv2
import torch
import time
import threading
from transformers import YolosForObjectDetection, YolosFeatureExtractor
import tkinter as tk
from PIL import Image, ImageTk

# List of categories from the dataset
categories = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat',
    'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt',
    'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette',
    'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle',
    'sequin', 'tassel'
]

# Items to check for
items_to_check = ['sweater', 'cardigan','coat', 'cape', 'glasses', 'hat', 'watch', 'belt', 'scarf'] #jacket

# Load the model and feature extractor
MODEL_NAME = "DatSplit/yolos-base-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
model_fashion = YolosForObjectDetection.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model_fashion = model_fashion.to('cuda')

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize tkinter for GUI
root = tk.Tk()
root.title("Self-service divest (security lane)")
root.geometry("600x400")
root.configure(bg="#003366")  # Dark blue background

# GUI Elements
detected_label = tk.Label(root, text="Detected Items:", font=("Helvetica", 16, "bold"), bg="#003366", fg="white")
detected_label.pack(pady=10)
items_text = tk.Label(root, text="", font=("Helvetica", 16), bg="#003366", fg="green")
items_text.pack(pady=10)
frame_label = tk.Label(root, bg="black")
frame_label.pack(pady=20)

# Detection tracker
detected_items_tracker = {}
last_detection_time = 0

def detect_items(frame):
    """Detect items using the model."""
    inputs = feature_extractor(images=frame, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
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
    return detected_items

def update_gui(detected_items):
    """Update the GUI with detected items."""
    current_time = time.time()
    for item in detected_items:
        detected_items_tracker[item] = current_time
    items_to_display = [
        item for item, last_time in detected_items_tracker.items()
        if current_time - last_time <= 20
    ]
    items_text.config(text=f"Please remove your {', '.join(items_to_display)}." if items_to_display else "No items detected.")
    items_text.config(fg="red" if items_to_display else "green")

def video_stream():
    """Capture frames and update the GUI."""
    global last_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 480))  # Reduce frame size
        if time.time() - last_detection_time >= 0.5:  # Process every 0.5 seconds
            detected_items = detect_items(frame_resized)
            update_gui(detected_items)
            last_detection_time = time.time()
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frame_image = ImageTk.PhotoImage(pil_image)
        frame_label.config(image=frame_image)
        frame_label.image = frame_image

# Run video stream in a separate thread
threading.Thread(target=video_stream, daemon=True).start()
root.mainloop()
cap.release()
