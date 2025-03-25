import onnxruntime
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from transformers import RTDetrImageProcessor
import cv2
CATS = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket',
        'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
        'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
        'tie', 'glove', 'watch', 'belt', 'leg warmer',
        'tights, stockings', 'sock', 'shoe', 'bag, wallet',
        'scarf', 'umbrella', 'hood', 'collar', 'lapel',
        'epaulette', 'sleeve', 'pocket', 'neckline',
        'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower',
        'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']


def run_onnx_inference(image_path):
    # Initialize the ONNX Runtime session

    onnx_path = Path("onnx_models/epoch_rtdetr_v2_r101.onnx")
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    feature_extractor = RTDetrImageProcessor.from_pretrained(
        f"PekingU/rtdetr_v2_r101vd")
    # Load and preprocess the image
    image = Image.open(image_path)
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Process image with feature extractor
    inputs = feature_extractor(
        images=image,
        return_tensors="pt",
    )
    pixel_values = inputs.pixel_values

    # Run inference
    ort_inputs = {
        'pixel_values': pixel_values.numpy()
    }

    # Print shape for debugging
    print(f"Input shape: {pixel_values.shape}")

    logits, pred_boxes = session.run(['logits', 'pred_boxes'], ort_inputs)

    # Post-process the outputs
    scores = torch.sigmoid(torch.from_numpy(logits))
    # Get predictions above threshold
    threshold = 0.3
    max_scores, pred_labels = scores.max(-1)
    mask = max_scores > threshold

    filtered_boxes = torch.from_numpy(pred_boxes)[mask]
    filtered_scores = max_scores[mask]
    filtered_labels = pred_labels[mask]

    orig_width, orig_height = image.size

    # Draw boxes on image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        # The model outputs coordinates in [center_x, center_y, width, height] format
        # Convert to [x1, y1, x2, y2] format
        center_x, center_y, width, height = box.tolist()

        # Convert normalized coordinates to pixel coordinates
        x1 = int((center_x - width/2) * orig_width)
        y1 = int((center_y - height/2) * orig_height)
        x2 = int((center_x + width/2) * orig_width)
        y2 = int((center_y + height/2) * orig_height)

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, orig_width))
        y1 = max(0, min(y1, orig_height))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label and score
        text = f"Class {CATS[label]}: {score:.2f}"
        cv2.putText(img, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    output_path = Path("output_detection.jpg")
    cv2.imwrite(str(output_path), img)
    print(f"Detection results saved to {output_path}")

    return filtered_boxes, filtered_scores, filtered_labels


if __name__ == "__main__":
    # Provide path to your test image
    image_path = "../../debug_folder/p7_008.png"
    boxes, scores, labels = run_onnx_inference(image_path)
    print("Detected Objects:")
    for box, score, label in zip(boxes, scores, labels):
        print(f"Label: {CATS[label]}, Score: {score:.2f}, Box: {box.tolist()}")
