from fashionfail.visualization.predictions import visualize_bbox_predictions
import argparse
import os
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bounding box predictions on images.")
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to the predictions file (e.g., .npy or .npz)')
    parser.add_argument('--images_folder', type=str, required=True,
                        help='Path to the folder containing images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save the processed images with bounding boxes')
    parser.add_argument('--score_threshold', type=float, default=0.2,
                        help='Score threshold for displaying bounding boxes')
    parser.add_argument('--model_type', type=str, default='amrcnn',
                        help='Model type (amrcnn, fformer, etc.)')
    parser.add_argument('--benchmark_dataset', type=str, default='fashionveil',
                        help='Benchmark dataset (fashionveil or other)')
    parser.add_argument('--output_file_name', type=str, default='bbox_predictions',
                        help='Name of the output file')
    args = parser.parse_args()

    predictions = np.load(args.predictions_path, allow_pickle=True)
    if isinstance(predictions, np.lib.npyio.NpzFile):
        predictions = predictions["data"]

    os.makedirs(args.output_folder, exist_ok=True)

    # Create output file path
    output_file = os.path.join(
        args.output_folder, f"{args.output_file_name}.pdf")

    visualize_bbox_predictions(
        predictions=predictions,
        model_type=args.model_type,
        img_folder=args.images_folder,
        score_threshold=args.score_threshold,
        out_path=output_file,
        benchmark_dataset=args.benchmark_dataset
    )


if __name__ == "__main__":
    main()
