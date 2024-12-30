import cv2 
import torch
from transformers import YolosForObjectDetection, YolosFeatureExtractor
import tkinter as tk
from tkinter import font
from tkinter import messagebox
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
items_to_check = ['sweater', 'cardigan', 'jacket', 'coat', 'cape', 'glasses', 'hat', 'watch', 'belt', 'scarf']

# Load the model and feature extractor
MODEL_NAME = "DatSplit/yolos-base-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
model_fashion = YolosForObjectDetection.from_pretrained(MODEL_NAME)
from tkinter import PhotoImage

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize tkinter for GUI
root = tk.Tk()
root.title("Self-service divest (security lane)")
root.geometry("600x400")
root.configure(bg="#003366")  # Dark blue background like PointFwd
# Open the image file and resize it
img = Image.open("pwd_logo.png")
img = img.resize((32, 32))

# Convert the image for use in Tkinter
img_tk = ImageTk.PhotoImage(img)

# Set the window icon
root.iconphoto(True, img_tk)

# Create a frame for the title bar
title_frame = tk.Frame(root, bg="#003366", height=50)
title_frame.pack(fill="x", side="top")




# Main content frame (below the title bar)
main_frame = tk.Frame(root, bg="#003366")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Label to display detected items
detected_label = tk.Label(main_frame, text="Detected Items:", font=("Helvetica", 16, "bold"), bg="#003366", fg="white")
detected_label.pack(padx=20, pady=20)

# Create a dynamic label to update detected items
items_text = tk.Label(main_frame, text="", font=("Helvetica", 16), bg="#003366", fg="green", anchor="w", justify="left")
items_text.pack(padx=20)

# Create a frame for showing the webcam feed (as an image)
frame_label = tk.Label(main_frame, bg="black")
frame_label.pack(padx=20, pady=20)

# Update the update_gui function
def update_gui(detected_items):
    """Update the GUI with the detected items."""
    items_str = ', '.join(detected_items)
    items_text.config(text=f"Please remove your {items_str}." if detected_items else "No items detected.")
    
    # Animate text color change for better user engagement
    items_text.config(fg="red" if detected_items else "green")

    # Update the webcam frame in the tkinter window
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Convert the PIL Image to a format that tkinter can handle
        frame_image = ImageTk.PhotoImage(pil_image)

        # Update the label with the new image
        frame_label.config(image=frame_image)
        frame_label.image = frame_image  # Keep a reference to the image object

def detect_items(frame):
    """Detect items using the model."""
    inputs = feature_extractor(images=frame, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model_fashion(**inputs)

    # Get predictions (bounding boxes, labels, and scores)
    target_sizes = torch.tensor([frame.shape[0:2]])  # Height, Width
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Check if the user is wearing any of the specified items
    detected_items = set()
    for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
        if score > 0.4:  # Only display detections with high confidence
            label = label.item()
            category_name = categories[label] if label < len(categories) else 'Unknown'
            
            # Add the detected item to the set if it matches any from the items_to_check
            if category_name.lower() in items_to_check:
                detected_items.add(category_name.lower())
    return detected_items

while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect items in the frame
    detected_items = detect_items(frame)

    # Update the GUI with detected items
    update_gui(detected_items)

    # Update the tkinter window
    root.update()

    # Exit if 'q' is pressed on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
root.quit()
root.destroy()
