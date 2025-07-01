import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # noqa
from transformers import RTDetrImageProcessor
import argparse
import torch
from loguru import logger
from rtdetr import rtdetr


def parse_args():

    parser = argparse.ArgumentParser(
        description='Export RT-DETR model to ONNX format')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="../saved_models/model_trained.ckpt",
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="rtdetr_v2_r101_fashionpedia.onnx",
        help='Output filename for the ONNX model'
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default="PekingU/rtdetr_v2_r101vd",
        help='Pretrained model name or path'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=46,
        help='Number of classes in the model'
    )
    return parser.parse_args()


def export_model_to_onnx(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = rtdetr.load_from_checkpoint(
        args.checkpoint,
        _cats=args.num_classes
    )
    model.eval()
    model.to(device)

    feature_extractor = RTDetrImageProcessor.from_pretrained(
        args.pretrained_model)
    save_directory = Path("onnx_models")
    save_directory.mkdir(exist_ok=True)

    batch_size = 1
    channels = 3
    height = 800
    width = 800
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_directory / args.output,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits', 'pred_boxes'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
            'logits': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        }
    )

    feature_extractor.save_pretrained(save_directory)
    logger.info(f"Model exported successfully to {save_directory}")


if __name__ == "__main__":
    args = parse_args()
    export_model_to_onnx(args)
