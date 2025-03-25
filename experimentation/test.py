from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection, YolosFeatureExtractor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from utils import fix_channels, visualize_predictions

# Load the model and feature extractor from Hugging Face
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
model = YolosForObjectDetection.from_pretrained(
    "DatSplit/yolos-base-fashionpedia")

image = Image.open(open("test4.png", "rb"))
image = fix_channels(ToTensor()(image))
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
visualize_predictions(image, outputs, threshold=0.3)
