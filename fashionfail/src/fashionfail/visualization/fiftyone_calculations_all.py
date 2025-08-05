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

    def calculate_comprehensive_metrics_by_confidence(dataset, gt_field, pred_field, confidence_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Calculate TP, FP, TN, FN for all classes and all occlusion levels
        across different confidence thresholds.
        """

        # Get all unique classes and occlusion levels
        all_classes = dataset.distinct(f"{gt_field}.detections.label")
        all_occlusion_levels = dataset.distinct(
            f"{gt_field}.detections.occlusion")

        print(f"Classes: {all_classes}")
        print(f"Occlusion levels: {all_occlusion_levels}")

        all_results = []

        for target_class in all_classes:
            for occlusion_level in all_occlusion_levels:
                print(f"\nProcessing {target_class} - {occlusion_level}")

                for conf_thresh in confidence_thresholds:
                    tp = fp = tn = fn = 0

                    # Iterate through ALL samples (not just filtered ones)
                    for sample in dataset:
                        # Check if sample has target class with specific occlusion level in GT
                        gt_detections = sample[gt_field].detections if sample[gt_field] else [
                        ]
                        gt_has_target = any(
                            det.label == target_class and det.occlusion == occlusion_level
                            for det in gt_detections
                        )

                        # Check if model predicted target class above confidence threshold
                        pred_detections = sample[pred_field].detections if sample[pred_field] else [
                        ]
                        pred_has_target = any(
                            det.label == target_class and det.confidence >= conf_thresh
                            for det in pred_detections
                        )

                        # Classification metrics
                        if gt_has_target and pred_has_target:
                            tp += 1  # Correctly detected
                        elif gt_has_target and not pred_has_target:
                            fn += 1  # Missed detection
                        elif not gt_has_target and pred_has_target:
                            fp += 1  # False alarm
                        else:  # not gt_has_target and not pred_has_target
                            tn += 1  # Correct rejection

                    total = tp + fp + tn + fn

                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / \
                        (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = (tp + tn) / total if total > 0 else 0

                    all_results.append({
                        'class': target_class,
                        'occlusion_level': occlusion_level,
                        'confidence_threshold': conf_thresh,
                        'TP': tp,
                        'FP': fp,
                        'TN': tn,
                        'FN': fn,
                        'Total_Samples': total,
                        'TP_%': (tp / total) * 100 if total > 0 else 0,
                        'FP_%': (fp / total) * 100 if total > 0 else 0,
                        'TN_%': (tn / total) * 100 if total > 0 else 0,
                        'FN_%': (fn / total) * 100 if total > 0 else 0,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'Accuracy': accuracy,
                        'GT_Positives': tp + fn,  # Total GT instances
                        'Pred_Positives': tp + fp,  # Total predictions
                    })

        return pd.DataFrame(all_results)

    def calculate_summary_by_class_occlusion(comprehensive_df, confidence_threshold=0.5):
        """
        Create summary statistics for each class-occlusion combination at a specific confidence threshold
        """

        filtered_df = comprehensive_df[comprehensive_df['confidence_threshold']
                                       == confidence_threshold]

        summary = []
        for _, row in filtered_df.iterrows():
            if row['GT_Positives'] > 0:  # Only include combinations that exist in GT
                summary.append({
                    'Class': row['class'],
                    'Occlusion_Level': row['occlusion_level'],
                    'GT_Count': row['GT_Positives'],
                    'Detection_Rate_%': row['Recall'] * 100,
                    'False_Positive_Rate_%': (row['FP'] / (row['FP'] + row['TN'])) * 100 if (row['FP'] + row['TN']) > 0 else 0,
                    'Precision_%': row['Precision'] * 100,
                    'F1_Score': row['F1']
                })

        return pd.DataFrame(summary).sort_values(['Class', 'Occlusion_Level'])

    # Add this to your main execution
    if __name__ == "__main__":
        # ... existing code ...

        # Calculate comprehensive metrics for all classes and occlusion levels
        print("\n" + "="*50)
        print("COMPREHENSIVE ANALYSIS - ALL CLASSES & OCCLUSION LEVELS")
        print("="*50)

        comprehensive_results = calculate_comprehensive_metrics_by_confidence(
            fo_dataset,
            "ground_truth_detections",  # Use correct field name
            "rfdetrl_predictions_FashionVeil",
            # Fewer thresholds for readability
            confidence_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
        )

        # Save comprehensive results
        comprehensive_results.to_csv(
            "comprehensive_detection_metrics.csv", index=False)
        print(f"\nSaved comprehensive results to comprehensive_detection_metrics.csv")

        # Generate summary at confidence 0.5
        summary_df = calculate_summary_by_class_occlusion(
            comprehensive_results, confidence_threshold=0.5)
        print(f"\nSUMMARY AT CONFIDENCE 0.5:")
        print(summary_df.round(2))

        # Save summary
        summary_df.to_csv("detection_summary_conf_0.5.csv", index=False)

        # Print specific insights for watch class
        watch_results = comprehensive_results[comprehensive_results['class'] == 'watch']
        print(f"\nWATCH DETECTION ACROSS OCCLUSION LEVELS (Confidence 0.5):")
        watch_05 = watch_results[watch_results['confidence_threshold'] == 0.5]
        for _, row in watch_05.iterrows():
            if row['GT_Positives'] > 0:
                print(f"{row['occlusion_level']}: {row['Recall']*100:.1f}% detection rate ({row['TP']}/{row['GT_Positives']} detected, {row['FP']} false positives)")
