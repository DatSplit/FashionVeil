from ultralytics import YOLO
import cv2
import time
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from utils import fix_channels, visualize_predictions

# Load YOLO model
yolo_model = YOLO("yolo11x.pt")

# Load YOLOS model
MODEL_NAME = "DatSplit/yolos-base-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
model_fashion = YolosForObjectDetection.from_pretrained(MODEL_NAME)

# Start video capture
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Run object detection
    results = yolo_model.predict(frame, imgsz=(1920, 1080), agnostic_nms=True)
    detections = results[0]

    for i, bbox in enumerate(detections.boxes.xyxy):
        cls_id = int(detections.boxes.cls[i])
        if cls_id == 0:  # Class 'person'
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox.cpu().numpy())
            cropped_person = frame[y1:y2, x1:x2]

            # Convert to PIL image
            image = Image.fromarray(cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB))
            image = fix_channels(ToTensor()(image))  # Fix channels if required

            # Run through YOLOS model
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model_fashion(**inputs)

            # Visualize predictions
            img = visualize_predictions(image, outputs, threshold=0.4)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
