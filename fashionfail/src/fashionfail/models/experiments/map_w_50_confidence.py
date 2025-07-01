import subprocess
from loguru import logger
from pathlib import Path
thresholds = [0.1 + (i * 0.1) for i in range(9)]

for thresh in thresholds:
    cmd = [
        "python", "/home/datsplit/model_development/fashionfail/src/fashionfail/models/predict_rfdetr.py",
        "--model_name", "rfdetr",
        "--image_dir", "/home/datsplit/model_development/FashionVeil_supercategories",
        "--out_dir", "predictions_fashionveil_all/",
        "--fashionveil_mapping", "True",
        "--confidence_threshold", str(thresh)
    ]
    logger.info(f"Running predictions with command: {' '.join(cmd)}")
    subprocess.run(cmd)

for pred_file in Path("/home/datsplit/model_development/fashionfail/src/predictions_fashionveil_all/").glob("*.npz"):
    logger.debug(f"Processing prediction file: {pred_file}")
    cmd = [
        "python", "/home/datsplit/model_development/fashionfail/src/fashionfail/models/evaluate_new.py",
        "--preds_path", str(pred_file),
        "--anns_path", "/home/datsplit/model_development/fashionveil_coco.json",
        "--eval_method", "Confidences",
        "--model_name", "rfdetr",
        "--iou_type", "bbox",
        "--benchmark_dataset", "fashionveil",
    ]
    logger.info(f"Running evaluation with command: {' '.join(cmd)}")
    subprocess.run(cmd)
