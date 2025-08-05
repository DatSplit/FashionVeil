"""
This will yield a mongodb instance at: "~/.fiftyone/"

Notes:
    Check out for more info about fiftyone:
    https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb#scrollTo=Jt2d_dlhW3Fv
    More on COCO evalution:
    - https://docs.voxel51.com/integrations/coco.html#coco-style-evaluation
    - https://docs.voxel51.com/tutorials/evaluate_detections.html#Evaluate-detections
"""

import argparse

import fiftyone as fo
import fiftyone.utils.coco as fouc
from loguru import logger
from fiftyone import ViewField as F
from fashionfail.utils import load_categories, load_fashionveil_categories
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Full path to the images directory.",
    )
    parser.add_argument(
        "--anns_dir",
        type=str,
        required=True,
        help="Full path to the bbox and masks annotations directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="FashionFail-test",
        help="Name of the dataset within fiftyone.",
    )

    return parser.parse_args()


def init_fo_dataset(image_dir, ann_path, name):
    # Import dataset by explicitly providing paths to the source media and labels
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_dir,
        labels_path=ann_path,
        name=name,
        label_field="ground_truth",
    )
    return dataset


# def init_fo_dataset(image_dir, ann_path, name):
#     # Create empty dataset
#     dataset = fo.Dataset(name)

#     # Add images
#     dataset.add_images_dir(image_dir)

#     # Add COCO labels WITH custom fields (including occlusion)
#     fouc.add_coco_labels(
#         dataset,
#         label_field="ground_truth",
#         labels_or_path=ann_path,
#         label_type="detections",
#     )

#     return dataset


def add_predictions(dataset, preds_path, preds_name: str):
    # And add model predictions
    fouc.add_coco_labels(
        dataset,
        label_field=preds_name,
        labels_or_path=preds_path,
        # classes=dataset.distinct("ground_truth_detections.detections.label"),
        categories=list(load_fashionveil_categories().values()),
        label_type="detections",
    )


if __name__ == "__main__":
    # Prediction paths of models
    # `amrcnn`
    # PREDS_AMRCNN_FF_TEST = "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/spinenet143-ff_test(filtered).json"
    # PREDS_AMRCNN_FP_VAL = (
    #     "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/"
    # )
    # # `amrcnn-R50`
    # PREDS_AMRCNNR50_FF_TEST = "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/r50fpn-ff_test(filtered).json"
    # PREDS_AMRCNNR50_FP_VAL = (
    #     "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/"
    # )
    # # `fformer`
    # PREDS_FFORMER_FF_TEST = "/home/rizavelioglu/work/repos/FashionFormer/outputs/fashionformer_swin_b_3x-ff-test(filtered).json"
    # PREDS_FFORMER_FP_VAL = "/home/rizavelioglu/work/repos/FashionFormer/outputs/"
    # # `fformer-R50`
    # PREDS_FFORMERR50_FF_TEST = "/home/rizavelioglu/work/repos/FashionFormer/outputs/fashionformer_r50_3x-ff-test(filtered).json"
    # PREDS_FFORMERR50_FP_VAL = "/home/rizavelioglu/work/repos/FashionFormer/outputs/"
    # `facere`
    # PREDS_FACERE_FF_TEST = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_base/facere_2_ff-test(filtered).json"
    # PREDS_FACERE_FP_VAL = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_base/"
    PREDS_RFDETRL_FV_ALL = "/home/datsplit/FashionVeil/fashionfail/src/predictions_fashionveil_all_rfdetrl/rfdetr_0.1-coco.json"
    PREDS_FFORMER_FV_ALL = "/home/datsplit/FashionVeil/fashionfail/predictions_fashionformer_swinb_fashionveil/fashionformer_swin_b_3x-fashionveil-coco.json"
    # `facere_plus`
    # PREDS_FACEREP_FF_TEST = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_plus_ff-test-coco.json"
    # PREDS_FACEREP_FP_VAL = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_plus_fp-val-coco.json"

    # Parse cli arguments
    args = get_cli_args()

    try:
        fo_dataset = fo.load_dataset(args.dataset_name)
    except ValueError:
        logger.info(
            f"{args.dataset_name} does not exist! Initializing one now...")
        fo_dataset = init_fo_dataset(
            image_dir=args.image_dir, ann_path=args.anns_dir, name=args.dataset_name
        )
        logger.info(f"Adding predictions...")
        # add_predictions(fo_dataset, PREDS_AMRCNN_FF_TEST, "preds-amrcnn")
        # add_predictions(fo_dataset, PREDS_AMRCNNR50_FF_TEST, "preds-amrcnnR50")
        # add_predictions(fo_dataset, PREDS_FFORMER_FF_TEST, "preds-fformer")
        # add_predictions(fo_dataset, PREDS_FFORMERR50_FF_TEST,
        #                 "preds-fformerR50")
        # add_predictions(fo_dataset, PREDS_FACERE_FP_VAL, "preds-facere")
        add_predictions(fo_dataset, PREDS_RFDETRL_FV_ALL,
                        "rfdetrl_predictions_FashionVeil_all")
        add_predictions(fo_dataset, PREDS_FFORMER_FV_ALL,
                        "fformer_predictions_FashionVeil")

    def evaluate_label_predictions(dataset: fo.Dataset, gt_field: str, pred_field: str, confidence_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], class_filter=None) -> pd.DataFrame:
        """
        args:
            dataset: fiftyone dataset object
            gt_field: ground truth field name (e.g. "ground_truth_detections")
            pred_field: predictions field name (e.g. "rfdetrl_predictions_FashionVeil")
            confidence_thresholds: list of confidence thresholds to evaluate
            class_name: optional class name (e.g. "watch") to evaluate only that specific class across all images
        returns:
            DataFrame with evaluation results containing detection rates, misrates etc.

        Calculate FP, FN, TP percentages for wearables detection.
        """
        result_df = []

        for confidence_threshold in confidence_thresholds:
            logger.info(
                f"Evaluating confidence threshold: {confidence_threshold}")

            count_gt_labels = 0
            count_predicted_labels = 0
            total_detected_labels = 0

            for image in dataset:
                gt_labels = image[gt_field].detections if image[gt_field] else [
                ]
                pred_detections = image[pred_field].detections if image[pred_field] else [
                ]

                if class_filter:
                    gt_labels = [
                        det for det in gt_labels if det.label == class_filter]
                    pred_detections = [
                        det for det in pred_detections if det.label == class_filter]

                count_gt_labels += len(gt_labels)

                predicted_labels = [
                    det for det in pred_detections if det.confidence >= confidence_threshold
                ]
                count_predicted_labels += len(predicted_labels)

                if len(predicted_labels) > 0 and len(gt_labels) > 0:
                    if class_filter:
                        total_detected_labels += min(len(gt_labels),
                                                     len(predicted_labels))
                    else:
                        gt_classes = set(det.label for det in gt_labels)
                        correct_class_predictions = [
                            pred for pred in predicted_labels
                            if pred.label in gt_classes
                        ]

                        for gt_class in gt_classes:
                            gt_count = sum(
                                1 for gt in gt_labels if gt.label == gt_class)
                            pred_count = sum(
                                1 for pred in correct_class_predictions if pred.label == gt_class)
                            total_detected_labels += min(gt_count, pred_count)

            detection_rate = total_detected_labels / \
                count_gt_labels if count_gt_labels > 0 else 0

            result_df.append({
                'confidence_threshold': confidence_threshold,
                'ground_truth_labels_count': count_gt_labels,
                'predicted_labels_count': count_predicted_labels,
                'correct_labels_predictions': total_detected_labels,
                'labels_detected_%': detection_rate * 100,  # detection rate
                # mistake rate
                'false_positives_%': ((count_predicted_labels - total_detected_labels) / count_predicted_labels * 100) if count_predicted_labels > 0 else 0,
                # missed wearables rate
                'false_negatives_%': ((count_gt_labels - total_detected_labels) / count_gt_labels * 100) if count_gt_labels > 0 else 0,
                'true_negatives_%': (1 - ((count_predicted_labels - total_detected_labels) / count_predicted_labels) if count_predicted_labels > 0 else 1)*100
            })

        return pd.DataFrame(result_df)


if __name__ == "__main__":

    detection_results = evaluate_label_predictions(
        fo_dataset,
        "ground_truth_detections",
        "rfdetrl_predictions_FashionVeil"
    )

    print("\nOVERALL OBJECT-LEVEL METRICS:")
    print(detection_results.round(2))

    print(f"\nKEY INSIGHTS AT CONFIDENCE 0.5:")
    obj_05 = detection_results[detection_results['confidence_threshold'] == 0.5].iloc[0]
    print(
        f"Object-level: {obj_05['labels_detected_%']:.1f}% detection rate")
    print(
        f"Total GT labels: {obj_05['ground_truth_labels_count']}, Total predictions: {obj_05['predicted_labels_count']}")

    colors = {
        "primary_dark": "#2E3440",
        "detection": "#F9A825",
        "false_pos": "#323a79",
        "false_neg": "#5EAADA",
        "true_neg": "#A3A3A3"
    }

    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        detection_results['confidence_threshold'],
        detection_results['labels_detected_%'],
        marker='o',
        label='True positive rate',
        color=colors["detection"],
        linewidth=2.5
    )

    ax.plot(
        detection_results['confidence_threshold'],
        detection_results['false_positives_%'],
        marker='s',
        label='False positive rate',
        color=colors["false_pos"],
        linewidth=2.5
    )

    ax.plot(
        detection_results['confidence_threshold'],
        detection_results['false_negatives_%'],
        marker='^',
        label='False negative rate',
        color=colors["false_neg"],
        linewidth=2.5
    )

    ax.plot(
        detection_results['confidence_threshold'],
        detection_results['true_negatives_%'],
        marker='d',
        label='True negative rate',
        color=colors["true_neg"],
        linewidth=2.5
    )

    ax.set_xlabel("Confidence threshold", fontsize=12,
                  color=colors["primary_dark"])
    ax.set_ylabel("Percentage (%)", fontsize=12, color=colors["primary_dark"])
    ax.set_title("Performance by confidence threshold (FashionVeil)",
                 fontsize=14, color=colors["primary_dark"])
    ax.tick_params(colors=colors["primary_dark"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')

    ax.set_xticks(detection_results['confidence_threshold'])

    plt.tight_layout()
    plt.savefig('confidence_threshold_performance.pdf',
                bbox_inches='tight', dpi=300)

    print(f"\nChart saved as 'confidence_threshold_performance.pdf'")
