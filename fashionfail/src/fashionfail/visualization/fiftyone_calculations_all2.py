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
import time
import re
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


# def init_fo_dataset(image_dir, ann_path, name):
#     # Import dataset by explicitly providing paths to the source media and labels
#     dataset = fo.Dataset.from_dir(
#         dataset_type=fo.types.COCODetectionDataset,
#         data_path=image_dir,
#         labels_path=ann_path,
#         name=name,
#         label_field="ground_truth",
#     )
#     return dataset


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

    # PREDS_FFORMER_FV_ALL = "/home/datsplit/FashionVeil/fashionfail/predictions_fashionformer_swinb_fashionveil/fashionformer_swin_b_3x-fashionveil-coco.json"
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
                        "rfdetrl_predictions_FashionVeil_new")
        # add_predictions(fo_dataset, PREDS_FFORMER_FV_ALL,
        #                 "fformer_predictions_FashionVeil")


def evaluate_label_predictions(
    dataset: fo.Dataset,
    gt_field: str,
    pred_field: str,
    confidence_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    class_filter=None
) -> pd.DataFrame:
    """
    Evaluates detection performance at the passenger level.
    """
    all_gt_labels = set(dataset.distinct(f"{gt_field}.detections.label"))
    print(f"Ground truth labels in dataset: {sorted(all_gt_labels)}")

    # Group by passenger
    passenger_groups = {}
    for sample in dataset:
        # Extract passenger ID from filename (e.g. "p20_001" → "p20")
        passenger_id = re.match(
            r"(p\d+)", sample.filename.split("/")[-1]).group(1)
        # print(passenger_id)
        if passenger_id not in passenger_groups:
            passenger_groups[passenger_id] = {
                "gt_labels": set(),
                "pred_labels": {}
            }

        # Ground truth labels
        gt_dets = sample[gt_field].detections if sample[gt_field] else []
        if class_filter:
            gt_dets = [d for d in gt_dets if d.label == class_filter]

        passenger_groups[passenger_id]["gt_labels"].update(
            [d.label for d in gt_dets])

        # Predictions (store per threshold)
        pred_dets = sample[pred_field].detections if sample[pred_field] else []
        if class_filter:
            pred_dets = [d for d in pred_dets if d.label == class_filter]

        for conf_thr in confidence_thresholds:
            if conf_thr not in passenger_groups[passenger_id]["pred_labels"]:
                passenger_groups[passenger_id]["pred_labels"][conf_thr] = set()
            passenger_groups[passenger_id]["pred_labels"][conf_thr].update(
                [d.label for d in pred_dets if d.confidence >= conf_thr]
            )

    # Compute metrics
    results = []
    remove_classes = {"top, t-shirt, sweatshirt"}
    for conf_thr in confidence_thresholds:
        TP = FP = FN = 0
        total_gt = total_pred = 0

        for passenger_id, data in passenger_groups.items():
            gt_labels = set(data["gt_labels"])
            pred_labels = set(data["pred_labels"][conf_thr])

            if "hood" not in gt_labels:
                gt_labels -= remove_classes
                pred_labels -= remove_classes

            total_gt += len(gt_labels)
            total_pred += len(pred_labels)

            if conf_thr == 0.5:
                print(
                    f"Passenger_id {passenger_id}, GT {gt_labels} ------------ PRED: {pred_labels}")

            TP += len(gt_labels & pred_labels)  # Intersection
            FP += len(pred_labels - gt_labels)
            FN += len(gt_labels - pred_labels)

        detection_rate = TP / total_gt if total_gt > 0 else 0

        results.append({
            "confidence_threshold": conf_thr,
            "ground_truth_labels_count": total_gt,
            "predicted_labels_count": total_pred,
            "correct_labels_predictions": TP,
            "labels_detected_%": detection_rate * 100,
            "false_positives_%": (FP / total_pred * 100) if total_pred > 0 else 0,
            "false_negatives_%": (FN / total_gt * 100) if total_gt > 0 else 0,
            "true_negatives_%": (1 - (FP / total_pred) if total_pred > 0 else 1) * 100
        })

    return pd.DataFrame(results)


def evaluate_label_predictions_by_occlusion(
    dataset: fo.Dataset,
    gt_field: str,
    pred_field: str,
    confidence_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    class_filter=None
) -> pd.DataFrame:
    """
    Evaluates detection performance per passenger and per occlusion level.
    """
    all_gt_labels = set(dataset.distinct(f"{gt_field}.detections.label"))
    print(f"Ground truth labels in dataset: {sorted(all_gt_labels)}")

    # Group by passenger and occlusion level
    passenger_groups = {}
    for sample in dataset:
        # Extract passenger ID from filename
        passenger_id = re.match(
            r"(p\d+)", sample.filename.split("/")[-1]).group(1)

        # Ground truth
        gt_dets = sample[gt_field].detections if sample[gt_field] else []
        if class_filter:
            gt_dets = [d for d in gt_dets if d.label == class_filter]

        # Predictions
        pred_dets = sample[pred_field].detections if sample[pred_field] else []
        if class_filter:
            pred_dets = [d for d in pred_dets if d.label == class_filter]

        for gt_det in gt_dets:

            occ_level = getattr(gt_det, "occlusion", None)
            key = (passenger_id, occ_level)

            if key not in passenger_groups:
                passenger_groups[key] = {
                    "gt_labels": set(),
                    "pred_labels": {thr: set() for thr in confidence_thresholds}
                }

            passenger_groups[key]["gt_labels"].add(gt_det.label)

            # Add predictions at all thresholds
            for conf_thr in confidence_thresholds:
                preds_in_sample = [
                    d.label for d in pred_dets
                    if d.confidence >= conf_thr
                ]
                passenger_groups[key]["pred_labels"][conf_thr].update(
                    preds_in_sample)

    # Compute metrics
    results = []
    remove_classes = {"top, t-shirt, sweatshirt"}
    for conf_thr in confidence_thresholds:
        for (passenger_id, occ_level), data in passenger_groups.items():
            gt_labels = set(data["gt_labels"])
            pred_labels = set(data["pred_labels"][conf_thr])

            if "hood" not in gt_labels:
                gt_labels -= remove_classes
                pred_labels -= remove_classes

            total_gt = len(gt_labels)
            total_pred = len(pred_labels)

            TP = len(gt_labels & pred_labels)
            FP = len(pred_labels - gt_labels)
            FN = len(gt_labels - pred_labels)

            detection_rate = TP / total_gt if total_gt > 0 else 0

            results.append({
                "passenger_id": passenger_id,
                "occlusion_level": occ_level,
                "confidence_threshold": conf_thr,
                "ground_truth_labels_count": total_gt,
                "predicted_labels_count": total_pred,
                "correct_labels_predictions": TP,
                "labels_detected_%": detection_rate * 100,
                "false_positives_%": (FP / total_pred * 100) if total_pred > 0 else 0,
                "false_negatives_%": (FN / total_gt * 100) if total_gt > 0 else 0,
                "true_negatives_%": (1 - (FP / total_pred) if total_pred > 0 else 1) * 100
            })

    return pd.DataFrame(results)


if __name__ == "__main__":

    detection_results = evaluate_label_predictions(
        fo_dataset,
        "ground_truth_detections",
        "rfdetrl_predictions_FashionVeil_new",
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
        "primary_dark": "#323a79",
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
        label='Detection rate',
        color=colors["detection"],
        linewidth=2.5
    )

    ax.plot(
        detection_results['confidence_threshold'],
        detection_results['false_positives_%'],
        marker='s',
        label='Mistake rate',
        color=colors["false_pos"],
        linewidth=2.5
    )

    # ax.plot(
    #     detection_results['confidence_threshold'],
    #     detection_results['false_negatives_%'],
    #     marker='^',
    #     label='Miss rate',
    #     color=colors["false_neg"],
    #     linewidth=2.5
    # )

    ax.set_xlabel("Confidence threshold", fontsize=12,
                  color=colors["primary_dark"])
    ax.set_ylabel("Percentage (%)", fontsize=12, color=colors["primary_dark"])
    ax.set_title("Average performance per passenger (FashionVeil)",
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
    # Save the dataset to disk

    import matplotlib.pyplot as plt
    import seaborn as sns
    detection_results = evaluate_label_predictions_by_occlusion(
        fo_dataset,
        "ground_truth_detections",
        "rfdetrl_predictions_FashionVeil_new",
    )
    # Assuming detection_results now has columns: passenger_id, occlusion_level, confidence_threshold, ...
    occlusion_order = [
        "No to slight occlusion",
        "Moderate occlusion",
        "Heavy occlusion",
        "Extreme occlusion"
    ]

    colors = {
        "primary_dark": "#323a79",
        "detection": "#F9A825",
        "false_pos": "#323a79",
        "false_neg": "#5EAADA",
        "true_neg": "#A3A3A3"
    }
    sns.set_style("white")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, occ_level in enumerate(occlusion_order):
        ax = axes[idx]

        occ_df = detection_results[
            detection_results["occlusion_level"] == occ_level
        ].groupby("confidence_threshold", as_index=False).mean(numeric_only=True)

        occ_df = occ_df.sort_values("confidence_threshold")

        ax.plot(
            occ_df['confidence_threshold'],
            occ_df['labels_detected_%'],
            marker='o',
            label='Detection rate',
            color=colors["detection"],
            linewidth=2.5
        )

        # ax.plot(
        #     occ_df['confidence_threshold'],
        #     occ_df['false_positives_%'],
        #     marker='s',
        #     label='Mistake rate',
        #     color=colors["false_pos"],
        #     linewidth=2.5
        # )

        ax.set_title(f"{occ_level}", fontsize=13, color=colors["primary_dark"])

        ax.set_xlabel("Confidence threshold", fontsize=11,
                      color=colors["primary_dark"])
        ax.set_ylabel("Percentage (%)", fontsize=11,
                      color=colors["primary_dark"])
        ax.tick_params(colors=colors["primary_dark"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xticks(occ_df['confidence_threshold'])

    # One legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('confidence_threshold_performance_by_occlusion.pdf',
                bbox_inches='tight', dpi=300)

    print("\nChart saved as 'confidence_threshold_performance_by_occlusion.pdf'")

    fo_dataset.persistent = False
    fo_dataset.save()

    session = fo.launch_app(fo_dataset, port=5151)
    session.wait()
