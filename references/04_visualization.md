<!-- Understanding the FashionFail repository:

In `predictions.py` filter out certain ground truth images, map pred to image.
Change load_categories() based on input_images_folder_name. -->

# Visualizations

Visualizing groups of predictions can show empirical evidence for the strengths and weaknesses of the different models.
To visualize predictions for an image set run the following commands:
```bash
cd visualizations
mkdir -p bbox_predictions
python visualize_bbox_predictions.py \
    --predictions_path "/home/datsplit/FashionVeil/fashionfail/predictions_fashionformer_swinb_fashionveil/fashionformer_swin_b_3x-fashionveil.npz" \
    --images_folder "/home/datsplit/FashionVeil/visualizations/smart_watch_images" \
    --output_folder "bbox_predictions" \
    --score_threshold "0.1" \
    --model_type "fformer" \
    --benchmark_dataset "fashionveil" \
    --output_file_name "bbox_preds_smart_watches" \
    --filter_single_class_name "watch"


python visualize_bbox_predictions.py \
    --predictions_path "/home/datsplit/FashionVeil/fashionfail/src/predictions_fashionveil_all_rfdetrl/rfdetr_0.1.npz" \
    --images_folder "/home/datsplit/FashionVeil/visualizations/smart_watch_images" \
    --output_folder "bbox_predictions" \
    --score_threshold "0.1" \
    --model_type "rfdetr" \
    --benchmark_dataset "fashionveil" \
    --output_file_name "bbox_preds_smart_watches_rfdetrl" \
    --filter_single_class_name "watch"
```
Explainability:
```bash
cd fashionfail/src/fashionfail/visualization
python3 fiftyone_test.py \
    --image_dir "/home/datsplit/FashionVeil/FashionVeil_all" \
    --anns_dir "/home/datsplit/FashionVeil/FashionVeil_all/FashionVeil_supercategories.json" \
    --dataset_name "predictions_FashionVeil"
```
Predicted bounding boxes of hood are quite often multiple times as large as the hood actually is for FasionFormer after visually the predictions in FiftyOne compared to fformer.

PCA in bounding box predictions.

```bash
python3 fiftyone_calculations_all2.py     --image_dir "/home/datsplit/FashionVeil/FashionVeil_all"     --anns_dir "/home/datsplit/FashionVeil/FashionVeil_all/FashionVeil_supercategories.json"     --dataset_name "predictions_FashionVeil"
```

---
<div style="display: flex; justify-content: space-between;">

   [Back](03_inference_and_evaluation.md)
</div>