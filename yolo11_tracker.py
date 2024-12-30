from ultralytics import YOLO
import cv2

video = cv2.VideoCapture("ski_solden.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video.release()

model = YOLO("yolo11x.pt")


results = model("ski_solden.mp4", save=True, show=True, imgsz=(width, height))

