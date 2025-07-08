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
    --score_threshold "0.2" \
    --model_type "fformer" \
    --benchmark_dataset "fashionveil" \
    --output_file_name "bbox_preds_smart_watches"
```


---
<div style="display: flex; justify-content: space-between;">

   [Back](03_inference_and_evaluation.md)
</div>