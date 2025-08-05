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
    # heavy_occlusion_view = fo_dataset.filter_labels(
    #     "ground_truth_detections",
    #     (F("occlusion") == "Heavy occlusion" & (F("label") == "watch")
    #      )
    # )

    # combined_view = fo_dataset.filter_labels(
    #     "ground_truth_detections",
    #     (F("occlusion") == "Heavy occlusion") & (F("label") == "watch")
    # ).match(
    #     (F("fformer_predictions_FashionVeil.detections").filter(F("label") == "watch").length() == 0) &
    #     (F("rfdetrl_predictions_FashionVeil_all.detections").filter(
    #         F("label") == "watch").length() == 0)
    # )

    combined_view = fo_dataset.filter_labels(
        "ground_truth_detections",
        (F("occlusion") == "Moderate occlusion") & (F("label") == "watch"))
    fo_dataset.persistent = True
    fo_dataset.save()

    import numpy as np
    import pandas as pd
    import json
    from collections import defaultdict

    def calculate_classification_metrics_by_confidence(dataset, gt_field, pred_field, target_class="watch", occlusion_level="Heavy occlusion"):
        """
        Calculate TP, FN percentages for classification (presence/absence of class) 
        across different confidence thresholds for occluded objects.
        Since we're filtering for samples that HAVE the target class, TN and FP don't apply.
        """

        # Filter for samples with heavily occluded target class
        filtered_view = dataset.filter_labels(
            gt_field,
            (F("occlusion") == occlusion_level) & (F("label") == target_class)
        )

        print(
            f"Total samples with {occlusion_level} {target_class}: {len(filtered_view)}")

        # Confidence thresholds to test
        confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []

        for conf_thresh in confidence_thresholds:
            tp = fn = 0

            for sample in filtered_view:
                # Ground truth: sample has occluded watch (always True for our filtered set)
                gt_has_target = True

                # Prediction: check if model predicted target class above confidence threshold
                pred_detections = sample[pred_field].detections if sample[pred_field] else [
                ]

                pred_has_target = any(
                    det.label == target_class and det.confidence >= conf_thresh
                    for det in pred_detections
                )

                # Classification metrics (presence/absence of class)
                if gt_has_target and pred_has_target:
                    tp += 1  # Correctly detected watch
                elif gt_has_target and not pred_has_target:
                    # Missed watch (either no prediction or low confidence)
                    fn += 1

            total = tp + fn  # Total samples with target class

            results.append({
                'confidence_threshold': conf_thresh,
                'TP': tp,
                'FN': fn,
                'Total_Samples': total,
                'TP_%': (tp / total) * 100 if total > 0 else 0,
                'FN_%': (fn / total) * 100 if total > 0 else 0,
                'Detection_Rate_%': (tp / total) * 100 if total > 0 else 0,
                'Miss_Rate_%': (fn / total) * 100 if total > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            })

        return pd.DataFrame(results)

    # Usage with your FiftyOne dataset
    results_df = calculate_classification_metrics_by_confidence(
        fo_dataset,
        "ground_truth_detections",
        "rfdetrl_predictions_FashionVeil",
        target_class="watch",
        occlusion_level="No to slight occlusion"
    )

    print("\nClassification metrics by confidence threshold:")
    print(results_df.round(2))

    # # Save results
    # results_df.to_csv(
    #     "watch_occlusion_classification_metrics.csv", index=False)

    # Print formatted results
    print(f"\nFormatted Results for Heavy occlusion watches:")
    for _, row in results_df.iterrows():
        print(
            f"Confidence {row['confidence_threshold']}: TP={row['TP_%']:.1f}%, FN={row['FN_%']:.1f}%")

    # session = fo.launch_app(combined_view, port=5151)
    # session.wait()
