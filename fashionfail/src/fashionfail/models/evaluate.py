from functools import lru_cache
from typing import Literal
import os
import json
from collections import Counter
import argparse

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fashionfail.models.cocoeval2 import COCOeval2
from fashionfail.models.prediction_utils import (
    bbox_conversion_formats,
    convert_preds_to_coco,
)
from fashionfail.utils import load_categories, load_fashionveil_categories, load_fashionpedia_divest_categories
import tempfile
from pathlib import Path
from fashionfail.models.prediction_utils import convert_preds_to_coco


def get_cli_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_path",
        type=str,
        required=True,
        help="Full path to the predictions file.",
    )
    parser.add_argument(
        "--anns_path",
        type=str,
        required=True,
        help="Full path to the annotations file.",
    )
    parser.add_argument(
        "--iou_type",
        type=str,
        choices=["bbox", "segm"],
        default="bbox",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        required=True,
        choices=["COCO", "COCO-extended", "all", "Confidences"],
        help="The name of the evaluation framework to be used, or `all` to run all eval methods.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=bbox_conversion_formats.keys(),
        help="The name of the model.",
    )

    parser.add_argument(
        "--occlusion_anns",
        type=bool,
        required=False,
        default=False,
        help="Boolean flag to indicate if occlusion level annotations should be used.",
    )

    parser.add_argument(
        "--benchmark_dataset",
        type=str,
        default="fashionveil",
        choices=["fashionveil", "fashionpedia", "fashionpedia_divest"],
        help="The benchmark dataset to use for evaluation. Default is 'fashionveil'.",
    )

    return parser.parse_args()


def print_per_class_metrics(coco_eval: COCOeval, return_results: bool = False) -> None | pd.DataFrame:
    logger.info("AP per class/category:")

    if cli_args.benchmark_dataset == "fashionveil":
        categories = load_fashionveil_categories()
    elif cli_args.benchmark_dataset == "fashionpedia":
        categories = load_categories()
    elif cli_args.benchmark_dataset == "fashionpedia_divest":
        categories = load_fashionpedia_divest_categories()
    cat_ids = coco_eval.params.catIds
    cat_names = [categories.get(cat_id) for cat_id in cat_ids]
    print("cat_ids:", cat_ids)
    print("cat_names:", cat_names)
    m_aps = []
    anns = coco_eval.cocoGt.dataset["annotations"]

    cat_ann_count = Counter(ann["category_id"] for ann in anns)

    for c in cat_ids:
        # [TxRxKxAxM]: A=0: area="all" & M=2: maxDets=100
        pr = coco_eval.eval["precision"][:, :, c, 0, 2]
        if len(pr[pr > -1]) == 0:
            m_ap = -1
        else:
            m_ap = np.mean(pr[pr > -1])
        m_aps.append(m_ap)
    # if cli_args.benchmark_dataset == "fashionveil":
    #     m_aps_shifted = m_aps[1:] + m_aps[:1]  # Hack to fix indexing.
    # else:
    #     m_aps_shifted = m_aps

    n_objs = [cat_ann_count.get(c, 0) for c in cat_ids]
    if cli_args.benchmark_dataset == "fashionveil":
        m_aps_shifted = m_aps[1:] + m_aps[:1]
        n_objs_shifted = n_objs[1:] + n_objs[:1]
    else:
        m_aps_shifted = m_aps
        n_objs_shifted = n_objs

    cats = pd.DataFrame(
        {"name": cat_names, "AP": m_aps_shifted, "#obj": n_objs_shifted})
    cats["name"] = cats["name"].str.slice(0, 15)

    if not return_results:
        display(cats)
    else:
        return cats


def print_tp_fp_fn_counts(coco_eval, iou_idx=0, area_idx=0, max_dets_idx=2):
    """
    Print a summary of metrics; TP, FP, FN counts, based on COCO evaluation results.

    Args:
        coco_eval (COCOeval2): An instance of the custom `COCOeval2` class, which is used as an alternative
            implementation to calculate and evaluate metrics that are not provided by the official COCOeval class.
        iou_idx (int, optional): Index for IoU threshold in [0.50, 0.05, 0.95]. Default is 0.
        area_idx (int, optional): Index for area range in ['all', 'small', 'medium', 'large']. Default is 0.
        max_dets_idx (int, optional): Index for maximum detections in [1, 10, 100]. Default is 2.

    Example:
        >>> print_tp_fp_fn_counts(coco_eval)
    """

    # Can't use `isinstance` because `coco_eval` is modified
    if coco_eval.__module__ != COCOeval2.__module__:
        logger.error(f"`coco_eval` object must be an object of {COCOeval2}!")
        return

    logger.info("TP,FP,FN counts per class/category:")

    print(
        f"Metrics @[",
        f"IoU={coco_eval.params.iouThrs[iou_idx]} |",
        f"area={coco_eval.params.areaRngLbl[area_idx]} |",
        f"maxDets={coco_eval.params.maxDets[max_dets_idx]} ]",
    )

    print("_" * 30)
    print(f"| {'cat':<2} | {'TP':<5} | {'FP':<5} | {'FN':<5} |")  # header
    print(f"|{'-' * 5}|{'-' * 7}|{'-' * 7}|{'-' * 7}|")  # separator

    total_tp, total_fp, total_fn = 0, 0, 0

    for catId in list(load_categories().keys()):
        num_tp = int(coco_eval.eval["num_tp"]
                     [iou_idx, catId, area_idx, max_dets_idx])
        num_fp = int(coco_eval.eval["num_fp"]
                     [iou_idx, catId, area_idx, max_dets_idx])
        num_fn = int(coco_eval.eval["num_fn"]
                     [iou_idx, catId, area_idx, max_dets_idx])

        print(f"| {catId:<3} | {num_tp:<5} | {num_fp:<5} | {num_fn:<5} |")

        total_tp += num_tp
        total_fp += num_fp
        total_fn += num_fn

    print(f"{'-' * 30}")
    print(f"{'Total':<5} | {total_tp:<5} | {total_fp:<5} | {total_fn:<5} |")


def compute_map_weighted(coco_eval, anns_path, cli_args, area_idx=0, max_dets_idx=2) -> None:
    logger.info("mAP & weighted mAP (main eval metric):")

    # Get class frequencies from the annotations file
    cat_freqs = calculate_class_frequencies(anns_path)
    logger.debug(f"Category frequencies: {cat_freqs.items()}")
    # mAP calculation
    map, w_map, w_map_50, w_map_75 = 0, 0, 0, 0
    for catId, catW in cat_freqs.items():
        map += np.nanmean(
            coco_eval.eval["precision"][:, :, catId, area_idx, max_dets_idx]
        )

        w_map += catW * np.nanmean(
            coco_eval.eval["precision"][:, :, catId, area_idx, max_dets_idx]
        )
        w_map_50 += catW * np.nanmean(
            coco_eval.eval["precision"][0, :, catId, area_idx, max_dets_idx]
        )
        w_map_75 += catW * np.nanmean(
            coco_eval.eval["precision"][5, :, catId, area_idx, max_dets_idx]
        )

    print(
        f"\n==== mAP & weighted-mAP ====",
        f"\nmAP     = {map / len(cat_freqs):.3f}",
        f"\nw-mAP   = {w_map:.3f}",
        f"\nw-mAP50 = {w_map_50:.3f}",
        f"\nw-mAP75 = {w_map_75:.3f}",
    )
    if cli_args.eval_method == "Confidences":
        confidence_threshold = cli_args.preds_path.split(
            "/")[-1].split("_")[1][:3]
        df_name = cli_args.model_name
        if not os.path.exists(f"{df_name}.csv"):
            df = pd.DataFrame(
                {"confidence_threshold": [confidence_threshold], "wmap50": [w_map_50]})
            df.to_csv(f"{df_name}.csv", index=False)
        else:
            df = pd.read_csv(f"{df_name}.csv")
            new_row = {
                "confidence_threshold": confidence_threshold, "wmap50": w_map_50}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(f"{df_name}.csv", index=False)
        logger.info(
            f"Saved mAP results to {df_name}.csv with confidence threshold {confidence_threshold} and w-mAP50 {w_map_50:.3f}."
        )

    # mAR calculation
    mar1, mar100, w_mar1, w_mar100 = 0, 0, 0, 0
    for catId, catW in cat_freqs.items():
        mar1 += np.nanmean(coco_eval.eval["recall"][:, catId, area_idx, 0])
        mar100 += np.nanmean(coco_eval.eval["recall"][:, catId, area_idx, 2])
        w_mar1 += catW * \
            np.nanmean(coco_eval.eval["recall"][:, catId, area_idx, 0])
        w_mar100 += catW * \
            np.nanmean(coco_eval.eval["recall"][:, catId, area_idx, 2])

    print(
        f"==== mAR & weighted-mAR ====",
        f"\nmAR1     = {mar1 / len(cat_freqs):.3f}",
        f"\nmAR100   = {mar100 / len(cat_freqs):.3f}",
        f"\nw-mAR1   = {w_mar1:.3f}",
        f"\nw-mAR100 = {w_mar100:.3f}",
    )


def compute_map_weighted_by_occlusion(coco_eval, anns_path, area_idx=0, max_dets_idx=2) -> None:
    """
    Calculate weighted mAP for different occlusion levels directly from COCO annotations.

    Args:
        coco_eval: COCO evaluation object (not used for per-occlusion eval)
        anns_path: Path to COCO annotations (with occlusion level in each annotation)
        area_idx: Area index (0=all, 1=small, 2=medium, 3=large)
        max_dets_idx: Maximum detections index
    """

    logger.info("Calculating mAP weighted by occlusion level:")

    with open(anns_path, 'r') as f:
        data = json.load(f)
        annotations = data.get('annotations', [])
        images = data.get('images', [])
        categories = data.get('categories', [])

    # Mapping: (image_id, annotation_id) -> occlusion_level
    occ_map = {}
    for ann in annotations:
        image_id = ann['image_id']
        ann_id = ann['id']
        occlusion_level = ann.get('occlusion', "Unknown")
        occ_map[(image_id, ann_id)] = occlusion_level

    logger.info(f"Loaded occlusion data for {len(occ_map)} annotations.")
    occlusion_levels = sorted(list(set(occ_map.values())))
    logger.info(f"Found occlusion levels: {occlusion_levels}")

    preds_path = getattr(coco_eval, 'preds_path', None)
    if preds_path is None:
        try:
            preds_path = cli_args.preds_path
        except Exception:
            logger.error(
                "Could not find the predictions file path for occlusion-level evaluation.")
            return

    for level in occlusion_levels:
        logger.info(f"\n=== Occlusion Level: {level} ===")

        filtered_annotations = [ann for ann in annotations if ann.get(
            'occlusion', "Unknown") == level]
        if not filtered_annotations:
            logger.info(f"No annotations found for occlusion level: {level}")
            continue
        cat_ann_count = Counter(ann['category_id']
                                for ann in filtered_annotations)
        image_ids = set(ann['image_id'] for ann in filtered_annotations)
        filtered_images = [img for img in images if img['id'] in image_ids]

        filtered_data = {
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': categories
        }

        other_level_anns = [ann for ann in annotations if ann.get(
            'occlusion', "Unknown") != level]
        other_level_boxes = {}
        for ann in other_level_anns:
            img_id = ann['image_id']
            bbox = ann['bbox']
            other_level_boxes.setdefault(img_id, []).append(bbox)

        def compute_iou(box1, box2):

            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            inter_w = max(0, xb - xa)
            inter_h = max(0, yb - ya)
            inter_area = inter_w * inter_h
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            if union_area == 0:
                return 0.0
            return inter_area / union_area

        with tempfile.TemporaryDirectory() as tmpdir:
            ann_file = os.path.join(tmpdir, f"anns_{level}.json")
            with open(ann_file, 'w') as f:
                json.dump(filtered_data, f)

            cat_freqs = calculate_class_frequencies(ann_file)
            coco_preds_file = preds_path.replace(
                Path(preds_path).suffix, "-coco.json")
            logger.info(f"Loading predictions from: {coco_preds_file}")
            with open(coco_preds_file, 'r') as pf:
                all_preds = json.load(pf)
            filtered_preds = []
            for p in all_preds:
                if p['image_id'] not in image_ids:
                    continue
                overlaps = False
                for other_box in other_level_boxes.get(p['image_id'], []):
                    if compute_iou(p['bbox'], other_box) > 0.5:
                        overlaps = True
                        break
                if not overlaps:
                    filtered_preds.append(p)
            pred_file = os.path.join(tmpdir, f"preds_{level}.json")

            with open(pred_file, 'w') as pf:
                json.dump(filtered_preds, pf)

            coco = COCO(ann_file)
            coco_dt = coco.loadRes(pred_file)
            coco_eval_level = COCOeval(
                coco, coco_dt, iouType=cli_args.iou_type)
            catIds_eval = coco_eval_level.params.catIds
            catId_to_local_idx = {catId: idx for idx,
                                  catId in enumerate(catIds_eval)}
            coco_eval_level.evaluate()
            coco_eval_level.accumulate()
            # Weighted mAP
            w_map, w_map_50, w_map_75 = 0, 0, 0
            w_mar, w_mar1, w_mar100 = 0, 0, 0
            map_values = {}
            mar_values = {}

            for catId, catW in cat_freqs.items():
                if catId not in catId_to_local_idx:
                    continue  # Skip categories not present in the current occlusion level
                local_idx = catId_to_local_idx[catId]
                precision = coco_eval_level.eval["precision"][:,
                                                              :, local_idx, area_idx, max_dets_idx]
                valid = precision > -1
                if np.any(valid):
                    map_value = np.mean(precision[valid])
                else:
                    map_value = 0.0
                map_values[catId] = map_value
                w_map += catW * map_value if not np.isnan(map_value) else 0.0

                precision_50 = coco_eval_level.eval["precision"][0,
                                                                 :, local_idx, area_idx, max_dets_idx]
                valid_50 = precision_50 > -1
                if np.any(valid_50):
                    ap_50 = np.mean(precision_50[valid_50])
                else:
                    ap_50 = 0.0
                w_map_50 += catW * ap_50 if not np.isnan(ap_50) else 0.0

                precision_75 = coco_eval_level.eval["precision"][5,
                                                                 :, local_idx, area_idx, max_dets_idx]
                valid_75 = precision_75 > -1
                if np.any(valid_75):
                    ap_75 = np.mean(precision_75[valid_75])
                else:
                    ap_75 = 0.0
                w_map_75 += catW * ap_75 if not np.isnan(ap_75) else 0.0

                recall = coco_eval_level.eval["recall"][:,
                                                        local_idx, area_idx, max_dets_idx]
                valid_r = recall > -1
                if np.any(valid_r):
                    mar_value = np.mean(recall[valid_r])
                else:
                    mar_value = 0.0
                mar_values[catId] = mar_value
                w_mar += catW * mar_value if not np.isnan(mar_value) else 0.0

                recall1 = coco_eval_level.eval["recall"][:,
                                                         local_idx, area_idx, 0]
                valid_r1 = recall1 > -1
                if np.any(valid_r1):
                    mar1 = np.mean(recall1[valid_r1])
                else:
                    mar1 = 0.0
                w_mar1 += catW * mar1 if not np.isnan(mar1) else 0.0

                recall100 = coco_eval_level.eval["recall"][:,
                                                           local_idx, area_idx, 2]
                valid_r100 = recall100 > -1
                if np.any(valid_r100):
                    mar100 = np.mean(recall100[valid_r100])
                else:
                    mar100 = 0.0
                w_mar100 += catW * mar100 if not np.isnan(mar100) else 0.0
            logger.info(
                f"\n==== mAP & weighted-mAP for {level} occlusion (in {len(filtered_annotations)} objects) ====")
            logger.info(f"w-mAP   = {w_map:.3f}")
            logger.info(f"w-mAP50 = {w_map_50:.3f}")
            logger.info(f"w-mAP75 = {w_map_75:.3f}")
            logger.info(f"w-mAR   = {w_mar:.3f}")
            logger.info(f"w-mAR1  = {w_mar1:.3f}")
            logger.info(f"w-mAR100= {w_mar100:.3f}")
            logger.info("\nPer-category AP for this occlusion level:")
            categories_dict = load_fashionveil_categories()

            # ToDo: fix this annoying +1/-1.
            incremented_dict = {k + 1: v for k, v in categories_dict.items()}

            for cat_id, ap in map_values.items():
                cat_name = incremented_dict.get(cat_id, "unknown")[:15]
                n_obj = cat_ann_count.get(cat_id, 0)
                logger.info(
                    f"{cat_id}: {cat_name:<15} - AP: {ap:.3f} - #obj: {n_obj}")

            logger.info("\nPer-category AR for this occlusion level:")
            for cat_id, ar in mar_values.items():
                cat_name = incremented_dict.get(cat_id, "unknown")[:15]
                n_obj = cat_ann_count.get(cat_id, 0)
                logger.info(
                    f"{cat_id}: {cat_name:<15} - AR: {ar:.3f} - #obj: {n_obj}")


@lru_cache
def calculate_class_frequencies(anns_path):
    # Load annotations
    coco_ann = COCO(anns_path)
    logger.debug(f"Loaded annotations from {anns_path}")
    # Define the FashionFail category ID's
    if cli_args.benchmark_dataset == "fashionpedia":
        logger.debug("Using Fashionpedia categories for evaluation.")
        cat_inds = list(set(range(27)) - {2, 12, 16, 19, 20})
        cat_weights = {}
    if cli_args.benchmark_dataset == "fashionpedia_divest":
        cat_inds = [id for id in coco_ann.getCatIds()]
        cat_weights = {cat_id: 0 for cat_id in cat_inds}
    if cli_args.benchmark_dataset == "fashionveil":
        cat_inds = [id-1 for id in coco_ann.getCatIds()]  # add id -1 back
        cat_weights = {cat_id: 0 for cat_id in cat_inds}

    # Retrieve number of samples per class
    for i in cat_inds:
        nb_samples = len(coco_ann.getImgIds(catIds=i))
        cat_weights[i] = nb_samples

    # Calculate total number of samples
    total_samples = sum(cat_weights.values())
    logger.info(
        f"Total number of samples in annotations: {total_samples}.")
    # Calculate the class frequencies
    for key, value in cat_weights.items():
        cat_weights[key] = value / total_samples
    logger.debug(f"Class frequencies: {cat_weights}")
    return cat_weights


def get_cocoeval(
    annotations_path: str,
    predictions_path: str,
    iou_type: Literal["bbox", "segm"] = "bbox",
    use_coco_eval2: bool = False,
):
    """
    Calculate COCO evaluation metrics for object detection or instance segmentation.

    Args:
        annotations_path (str): The file path to the ground truth annotations in COCO format.
        predictions_path (str): The file path to the prediction results in COCO format.
        iou_type (str): The type of intersection over union (IoU) to use for evaluation.
            Can be either "bbox" for bounding box IoU or "segm" for segmentation IoU. Default is "bbox".
        use_coco_eval2 (bool): If True, use a custom implementation (COCOeval2) to compute evaluation metrics,
            including TP (True Positives), FP (False Positives), and FN (False Negatives) counts.
            If False, use the standard COCOeval. Default is False.

    Returns:
        coco_eval: A COCO evaluation object containing computed metrics and results.

    Examples:
        Run official evalution and get access to 'eval' dict including metrics; 'precision',' recall', 'scores'.

        >>> coco_eval = get_cocoeval(annotations_path, predictions_path, iou_type="bbox", use_coco_eval2=False)

        Run customized evalution and get access to 'eval' dict including metrics; "num_tp", "num_fp", "num_fn",
        "scores_tp", "scores_fp" alongside 'precision',' recall', 'scores'.

        >>> coco_eval = get_cocoeval(annotations_path, predictions_path, iou_type="bbox", use_coco_eval2=True)

    """
    # Load GT annotations
    coco = COCO(annotations_path)

    # Load predictions (dt)
    coco_dt = coco.loadRes(predictions_path)

    # Use own implementation if specified which returns TP,FP,FN counts
    if use_coco_eval2:
        coco_eval = COCOeval2(coco, coco_dt, iouType=iou_type)
    else:
        coco_eval = COCOeval(coco, coco_dt, iouType=iou_type)

    if cli_args.benchmark_dataset == "fashionveil":
        logger.info("Using FashionVeil categories for evaluation.")
        coco_eval.params.catIds = list(load_fashionveil_categories().keys())
    elif cli_args.benchmark_dataset == "fashionpedia":
        coco_eval.params.catIds = list(load_categories().keys())
    elif cli_args.benchmark_dataset == "fashionpedia_divest":
        coco_eval.params.catIds = list(
            load_fashionpedia_divest_categories().keys())

    coco_eval.evaluate()
    coco_eval.accumulate()

    return coco_eval


def eval_with_coco(args, use_extended_coco: bool = False) -> None:

    logger.info(args.model_name)
    preds_path = convert_preds_to_coco(
        preds_path=args.preds_path, anns_path=args.anns_path, model_name=args.model_name
    )

    # Run evaluation and print results
    coco_eval = get_cocoeval(
        annotations_path=args.anns_path,
        predictions_path=preds_path,
        iou_type=args.iou_type,
        use_coco_eval2=use_extended_coco,
    )
    coco_eval.summarize()
    print_per_class_metrics(coco_eval)
    if use_extended_coco:
        print_tp_fp_fn_counts(coco_eval)
    if args.occlusion_anns:
        logger.info(
            "Calculating the metrics by occlusion level."
        )
        compute_map_weighted_by_occlusion(
            coco_eval,
            anns_path=args.anns_path,
        )
    else:
        compute_map_weighted(
            coco_eval, anns_path=args.anns_path, cli_args=args)


if __name__ == "__main__":
    cli_args = get_cli_args()

    if cli_args.eval_method == "COCO":
        eval_with_coco(cli_args)
    elif cli_args.eval_method == "COCO-extended":
        eval_with_coco(cli_args, use_extended_coco=True)
    elif cli_args.eval_method == "all":
        logger.info("=" * 10 + "Evaluating with official COCOeval" + "=" * 10)
        eval_with_coco(cli_args)
        logger.info("=" * 10 + "Evaluating with extended COCOeval" + "=" * 10)
        eval_with_coco(cli_args, use_extended_coco=True)
    elif cli_args.eval_method == "Confidences":
        eval_with_coco(cli_args)
    else:
        logger.error(
            f"`eval_method` must be one of ['COCO', 'COCO-extended', 'all', 'Confidences'], but passed: "
            f"{cli_args.eval_method}."
        )
