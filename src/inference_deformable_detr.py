from PIL import Image
import torch
from transformers import DeformableDetrImageProcessor, RTDetrImageProcessor
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from src.utils import fix_channels, visualize_predictions
from src.deformable_detr import DeformableDetrFashionpedia
from src.rtdetr import rtdetr
from loguru import logger
# Path to your saved checkpoint
# adjust to your actual checkpoint filename
CHECKPOINT_PATH = "saved_models/rtdetr-epoch=05-validation_loss=7.11.ckpt"


def load_model_from_checkpoint(checkpoint_path, num_classes=46):
    """Load the DeformableDetr model from checkpoint"""
    # Initialize model with dummy values - the checkpoint will override these
    model = rtdetr.load_from_checkpoint(
        checkpoint_path,
        _cats=num_classes
    )
    model.eval()  # Set to evaluation mode
    return model


def predict_image(model, image_path, threshold=0.3):
    """Make predictions on an image using the loaded model"""
    # Initialize the feature extractor
    feature_extractor = RTDetrImageProcessor()

    # Load and preprocess the image
    image = Image.open(image_path)
    image_tensor = fix_channels(ToTensor()(image))

    # Process for model input
    inputs = feature_extractor(
        [image_tensor], return_tensors='pt')

    # Move to same device as model
    device = next(model.parameters()).device

    expected_keys = ['pixel_values']  # Only keep the keys your model accepts
    filtered_inputs = {k: v.to(device)
                       for k, v in inputs.items() if k in expected_keys}

    # Make prediction
    with torch.no_grad():
        outputs = model(**filtered_inputs)
    # Visualize predictions
    visualize_predictions(image_tensor, outputs, threshold=0.4)

    return outputs


if __name__ == "__main__":
    # Load model from checkpoint
    model = load_model_from_checkpoint(CHECKPOINT_PATH)

    # Run prediction on a test image
    test_image_path = "p2_002.png"  # Change to your test image
    outputs = predict_image(model, test_image_path)

    # Print predictions
    print(f"Prediction scores: {outputs.logits.max(dim=-1)}")
