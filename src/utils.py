# pylint: disable=missing-module-docstring
from typing import List, Tuple

import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

CATS = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket',
        'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
        'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
        'tie', 'glove', 'watch', 'belt', 'leg warmer',
        'tights, stockings', 'sock', 'shoe', 'bag, wallet',
        'scarf', 'umbrella', 'hood', 'collar', 'lapel',
        'epaulette', 'sleeve', 'pocket', 'neckline',
        'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower',
        'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def id_to_text(i: int) -> str:
    """
    Convert index to category label.
    :param i: Index
    :return: Category label
    """
    return CATS[i]


def idx_to_text(indexes: List[int]) -> List[str]:
    """
    Converts an index into a category label.
    :param indexes: List of indexes
    :return: List of category labels
    """
    labels = []
    for i in indexes:
        labels.append(CATS[i])
    return labels


def fix_channels(t: torch.Tensor) -> Image.Image:
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images).
    In order to let the images have only 3 channels. 
    I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)


def xyxy_to_xcycwh(box: torch.Tensor) -> torch.Tensor:
    """
    Boxes in images may have the format (x1, y1, x2, y2) and we may need the format (center of x, center of y, width, height).
    :param box: Tensor-like box with format (x1, y1, x2, y2)
    :return: Tensor-like box with format (center of x, center of y, width, height)
    """
    x1, y1, x2, y2 = box.unbind(dim=1)
    width = x2 - x1
    height = y2 - y1
    xc = x1 + width * 0.5
    yc = y1 + height * 0.5
    b = [xc, yc, width, height]
    return torch.stack(b, dim=1)


def cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Boxes in images may have the format (center of x, center of y, width, height) and we may need the format (x1, y1, x2, y2).
    :param x: Tensor-like box with format (center of x, center of y, width, height)
    :return: Tensor-like box with format (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    b = [x1, y1, x2, y2]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center format to corner format.
    :param x: Tensor-like bounding box with format (center of x, center of y, width, height)
    :return: Tensor-like bounding box with format (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def plot_results(pil_img: Image.Image, prob: torch.Tensor, boxes: torch.Tensor) -> None:
    """
    Plot the results on the image.
    :param pil_img: PIL Image
    :param prob: Tensor-like probabilities
    :param boxes: Tensor-like bounding boxes
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, id_to_text(cl), fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    plt.savefig("image.png")


def visualize_predictions(image: Image.Image, outputs: torch.Tensor, threshold: float = 0.8) -> None:
    """
    Visualize the predictions on the image.
    :param image: PIL Image
    :param outputs: Model outputs
    :param threshold: Confidence threshold
    """
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(
        outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled)
