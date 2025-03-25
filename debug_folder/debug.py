import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
CATS = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket',
        'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
        'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
        'tie', 'glove', 'watch', 'belt', 'leg warmer',
        'tights, stockings', 'sock', 'shoe', 'bag, wallet',
        'scarf', 'umbrella', 'hood', 'collar', 'lapel',
        'epaulette', 'sleeve', 'pocket', 'neckline',
        'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower',
        'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

indices = [8, 24, 23,  1, 33, 23]

for index in indices:
    print(CATS[index])


# ...existing code...


def visualize_predictions(image_path, predictions, output_path=None, score_threshold=0.3):
    """
    Visualize bounding boxes on the image.

    Args:
        image_path: Path to the image file or directory containing the image
        predictions: Dictionary with keys 'boxes', 'classes', 'scores'
        output_path: Path to save the output image. If None, displays the image.
        score_threshold: Only show predictions with score above this threshold
    """
    # If image_path is a directory, look for the image file there
    if os.path.isdir(image_path):
        img_file = predictions.get('image_file', '')
        image_path = os.path.join(image_path, img_file)

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    # Get predictions
    boxes = predictions.get('boxes', [])
    classes = predictions.get('classes', [])
    scores = predictions.get('scores', [])

    # Draw each box
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255)]

    for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
        if score < score_threshold:
            continue

        # Get coordinates
        # Change from:
        x1, y1, x2, y2 = box

        # To:
        y1, x1, y2, x2 = box

        # Get color (cycle through colors)
        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        class_name = CATS[class_id]
        label = f"{class_name}: {score:.2f}"
        draw.rectangle([x1, y1, x1 + len(label) * 8, y1 + 20], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

    # Save or display
    if output_path:
        image.save(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        image.show()


# 2. For the RT-DETR model
rtdetr_predictions = {
    'image_file': 'f43728a7877b2a6b6ca5ebac30672d9e.jpg',
    'boxes': np.array([
        [367.25958, 250.76346, 393.60175, 384.61172],
        [368.01907, 245.33551, 501.43906, 384.4093],
        [368.01727, 229.58714, 704.4681, 397.89807],
        [583.0536, 151.51076, 719.28015, 252.40501],
        [879.889, 343.3959, 983.362, 389.10672],
        [486.95337, 227.0058, 706.0482, 398.82703],
        [367.4392, 250.20294, 399.3794, 383.6789],
        [870.5521, 190.26111, 979.50366, 242.13629]
    ], dtype=np.float32),
    'classes': np.array([33, 1, 10, 24, 23, 8, 33, 23]),
    'scores': np.array([
        0.7332024, 0.7409887, 0.40814435, 0.9644041, 0.94793785,
        0.83658105, 0.57378054, 0.93983287
    ], dtype=np.float32)
}

# Set the path to your images directory
images_dir = "./debug"


visualize_predictions(images_dir, rtdetr_predictions,
                      "f43728a7877b2a6b6ca5ebac30672d9e.jpg")
