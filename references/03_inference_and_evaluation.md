# Inference and evaluation

> [!IMPORTANT]
> The FashionFail repository is utilized for inferencing and evaluation, credits to [Velioglu et at.](https://github.com/rizavelioglu/fashionfail)

For inferencing on the benchmark datasets, we utilize a modified version of the repository of Velioglu et al. [FashionFail repository](https://github.com/rizavelioglu/fashionfail/tree/main).

We compare `RT-DETR`, `RF-DETR-B`, and `RF-DETR-L` with the state-of-the-art FashionFormer model, and with the Attribute Mask R-CNN and Facere models on the `Fashionpedia` and `FashionVeil` datasets.

Table modified from [FashionFail - Inference](https://github.com/rizavelioglu/fashionfail/blob/main/references/04_inference.md?plain=1).

| Model name        | Description                                                                 |       Backbone       | download                                                |
| ----------------- | --------------------------------------------------------------------------- | :------------------: | :------------------------------------------------------ |
| `amrcnn-spine`    | Attribute Mask-RCNN model released with [Fashionpedia paper][paper_amrcnn]. |     SpineNet-143     | [ckpt][amrcnn_spine_ckpt] \| [config][amrcnn_spine_cfg] |
| `fformer-swin`    | Fashionformer model released by [Fashionformer paper][paper_fformer].       |      Swin-base       | [pth][models_fformer]                                   |
| `amrcnn-r50-fpn`  | Attribute Mask-RCNN model released with [Fashionpedia paper][paper_amrcnn]. |     ResNet50-FPN     | [ckpt][amrcnn_r50_ckpt] \| [config][amrcnn_r50_cfg]     |
| `fformer-r50-fpn` | Fashionformer model released by [Fashionformer paper][paper_fformer].       |     ResNet50-FPN     | [pth][models_fformer]                                   |
| `facere`          | Mask R-CNN based model trained on `Fashionpedia-train`.                     |     ResNet50-FPN     | [onnx][facere_onnx]                                     |
| `RT-DETR`         | `RT-DETR` model finetuned on `FashionFail-train`.                           | ResNet50 / ResNet101 | available on request                                    |
| `RF-DETR-B`       | `RF-DETR-B` model finetuned on `FashionFail-train`.                         |        DinoV2        | available on request                                    |
| `RF-DETR-L`       | `RF-DETR-L` model finetuned on `FashionFail-train`.                         |        DinoV2        | available on request                                    |

For fair evaluation we make predictions on the same images as done in `FashionFail`.
This dataset can be constructed by running `python3 fashionfail/data/make_fashionpedia.py`.
This will create a folder in your `~/user/.cache` directory. The exact same `train/val` split is used when downloading our `Fashionpedia for RF-DETR` dataset.

### Inference on RT-DETR
##### Fashionpedia-test
For RT-DETR (ResNet50 and ResNet101 backbones) the results on `Fashionpedia-test` are noted in Table 7.1, the two code blocks below show how to calculate this.


The script below runs predictions on all images in the `image_dir` and outputs it to the `out_dir` as a `.npz` file.
```bash
python fashionfail/models/predict_rtdetr.py \
    --model_name "rtdetr" \
    --image_dir "DIRECTORY_OF_FASHIONPEDIA_TEST_IMAGES" \ # Usually located at ~/user/.cache/fashionpedia/images/test
    --out_dir "predictions_fashionpedia_test_rtdetr_ResNet50/" 
```
Now the predictions (`preds_path`) can be evaluated based on the ground truth bounding boxes and class labels (`anns_path`) on the 5 metrics mentioned in the report on the `fashionpedia-test` benchmark dataset.

```bash
python fashionfail/models/evaluate.py \
    --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionpedia_test_rtdetr_ResNet50/rtdetr.npz"
    --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/user/.cache/fashionpedia/instances_attributes_val2020.json
    --eval_method "COCO" \
    --model_name "rtdetr" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionpedia"
```

##### FashionVeil
For RT-DETR the results on `FashionVeil` are noted in Table 7.2.

The three code blocks below show how to calculate this for the `RT-DETR` model.

```bash
    python fashionfail/models/predict_models.py \
        --model_name "rtdetr" \
        --image_dir "DIRECTORY_OF_FASHIONVEIL_IMAGES" \ # Usually located at
        --out_dir "predictions_fashionveil_all_rtdetr_resnet50/" \
        --fashionveil_mapping True
```
The below block evaluates the predictions on the whole `FashionVeil` dataset.
```bash
    python fashionfail/models/evaluate.py \
        --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionveil_all_rtdetr_resnet50/rtdetr.npz
        --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/SOME_DIRECTORIES/fashionveil_coco.json
        --eval_method "COCO" \
        --model_name "rtdetr" \
        --iou_type "bbox" \
        --benchmark_dataset "fashionveil"
```

### Inference on RF-DETR

##### Fashionpedia-test
For RF-DETR (Base (B) and Large (L)) the results on `Fashionpedia-test` are noted in Table 7.1, the two code blocks below show how to calculate this.


The script below runs predictions on all images in the `image_dir` and outputs it to the `out_dir` as a `.npz` file.
```bash
python fashionfail/models/predict_rfdetr.py \
    --onnx_path "ONNX_FILE_LOCATION" \
    --model_name "rfdetr" \
    --image_dir "DIRECTORY_OF_FASHIONPEDIA_TEST_IMAGES" \ # Usually located at ~/user/.cache/fashionpedia/images/test
    --out_dir "predictions_fashionpedia_test_rdetrb/" 
```
Now the predictions (`preds_path`) can be evaluated based on the ground truth bounding boxes and class labels (`anns_path`) on the 5 metrics mentioned in the report on the `fashionpedia-test` benchmark dataset.

```bash
python fashionfail/models/evaluate.py \
    --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionpedia_test_rdetrb/rfdetr.npz"
    --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/user/.cache/fashionpedia/instances_attributes_val2020.json
    --eval_method "COCO" \
    --model_name "rfdetr" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionpedia"
```

##### FashionVeil
For RF-DETR the results on `FashionVeil` are noted in Table 7.2 and the occlusion specific performance is noted in Table 7.3 and Figure 7.1.

The three code blocks below show how to calculate this for the `RF-DETR` models.

```bash
    python fashionfail/models/predict_rfdetr.py \
        --onnx_path "ONNX_FILE_LOCATION" \
        --model_name "rfdetr" \
        --image_dir "DIRECTORY_OF_FASHIONVEIL_IMAGES" \ # Usually located at
        --out_dir "predictions_fashionveil_all_rfdetrb/" \
        --fashionveil_mapping True
```
The below block evaluates the predictions on the whole `FashionVeil` dataset.
```bash
    python fashionfail/models/evaluate.py \
        --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionveil_all_rfdetrb/rfdetr.npz
        --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/SOME_DIRECTORIES/fashionveil_coco.json
        --eval_method "COCO" \
        --model_name "rfdetr" \
        --iou_type "bbox" \
        --benchmark_dataset "fashionveil"
```

The below block evaluates the predictions on the `FashionVeil` dataset per occlusion level.
```bash
    python fashionfail/models/evaluate.py \
        --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionveil_all_rfdetrb/rfdetr.npz
        --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/SOME_DIRECTORIES/fashionveil_coco.json
        --eval_method "COCO" \
        --model_name "rfdetr" \
        --iou_type "bbox" \
        --occlusion_anns True \
        --benchmark_dataset "fashionveil"
```

RF-DETR was also fine-tuned on `FashionVeil-train/val` and evaluated on `FashionVeil-test` for all occlusion levels, which is shown in Table 7.4 and Figure 7.2:

```bash
python fashionfail/models/predict_rfdetr.py \
    --onnx_path "ONNX_FILE_LOCATION" \
    --model_name "rfdetr" \
    --image_dir "DIRECTORY_OF_FASHIONVEIL_TEST" \
    --out_dir "predictions_fashionveil_test_finetuned_rfdetrl/" \
    --fashionveil_mapping True
```

```bash
python fashionfail/models/evaluate.py \
    --preds_path "LOCATION_OF_PREDICTION_FILE" \
    --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # fashionveil_test.json
    --eval_method "COCO" \
    --model_name "rfdetr" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionveil"

python fashionfail/models/evaluate.py \
    --preds_path "LOCATION_OF_PREDICTION_FILE" \
    --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # fashionveil_test.json
    --eval_method "COCO" \
    --model_name "rfdetr" \
    --occlusion_anns True \
    --iou_type "bbox" \
    --benchmark_dataset "fashionveil"
```

The per category metric is evaluated by default, so we used those for the radar plots in Figure 7.3 and Figure 7.4.






















### Inference on Facere
##### Fashionpedia-test
For facere the results on `Fashionpedia-test` are noted in Table 7.1, the two code blocks below show how to calculate this.


The script below runs predictions on all images in the `image_dir` and outputs it to the `out_dir` as a `.npz` file.
```bash
python fashionfail/models/predict_models.py \
    --model_name "facere_base" \
    --image_dir "DIRECTORY_OF_FASHIONPEDIA_TEST_IMAGES" \ # Usually located at ~/user/.cache/fashionpedia/images/test
    --out_dir "predictions_fashionpedia_test_facere_base/" 
```
Now the predictions (`preds_path`) can be evaluated based on the ground truth bounding boxes and class labels (`anns_path`) on the 5 metrics mentioned in the report on the `fashionpedia-test` benchmark dataset.

```bash
python fashionfail/models/evaluate.py \
    --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionpedia_test_facere_base/facere_base.npz"
    --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/user/.cache/fashionpedia/instances_attributes_val2020.json
    --eval_method "COCO" \
    --model_name "facere" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionpedia"
```

##### FashionVeil
For facere the results on `FashionVeil` are noted in Table 7.2 and the occlusion specific performance is noted in Table 7.3 and Figure 7.1.

The three code blocks below show how to calculate this for the `facere` model.

```bash
    python fashionfail/models/predict_models.py \
        --model_name "facere_base" \
        --image_dir "DIRECTORY_OF_FASHIONVEIL_IMAGES" \ # Usually located at
        --out_dir "predictions_fashionveil_all_facere_base/" \
        --fashionveil_mapping True
```
The below block evaluates the predictions on the whole `FashionVeil` dataset.
```bash
    python fashionfail/models/evaluate.py \
        --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionveil_all_facere_base/facere_base.npz
        --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/SOME_DIRECTORIES/fashionveil_coco.json
        --eval_method "COCO" \
        --model_name "facere" \
        --iou_type "bbox" \
        --benchmark_dataset "fashionveil"
```

The below block evaluates the predictions on the `FashionVeil` dataset per occlusion level.
```bash
    python fashionfail/models/evaluate_new.py \
        --preds_path "LOCATION_OF_PREDICTION_FILE" \ # Usually located at ~/SOME_DIRECTORIES/predictions_fashionveil_all_facere_base/facere_base.npz
        --anns_path "LOCATION_OF_GT_ANNOTATIONS_FILE" \ # Usually located at ~/SOME_DIRECTORIES/fashionveil_coco.json
        --eval_method "COCO" \
        --model_name "facere" \
        --iou_type "bbox" \
        --occlusion_anns True \
        --benchmark_dataset "fashionveil"
```


### Inference on Attribute Mask R-CNN [[paper]][paper_amrcnn] [[code]][code_amrcnn]

> [!IMPORTANT]
> The following content is adapted from [Velioglu et at.](https://github.com/rizavelioglu/fashionfail):

Create and activate the conda environment:
```bash
conda create -n amrcnn python=3.9
conda activate amrcnn
```

Install dependencies:
```bash
pip install tensorflow-gpu==2.11.0 Pillow==9.5.0 pyyaml opencv-python-headless tqdm pycocotools
```

Clone the repository, navigate to the `detection` directory and download the models:
```bash
cd /change/dir/to/fashionfail/repo/
git clone https://github.com/jangop/tpu.git
cd tpu
git checkout 85b65b6
cd models/official/detection
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz --output fashionpedia-spinenet-143.tar.gz
tar -xf fashionpedia-spinenet-143.tar.gz
curl https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz
tar -xf fashionpedia-r50-fpn.tar.gz
```

Finally, inference can be run with:
```bash
cd some_path/fashionfail/tpu/models/official/detection
python inference_fashion.py \
    --model="attribute_mask_rcnn" \
    --config_file="projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml" \
    --checkpoint_path="fashionpedia-spinenet-143/model.ckpt" \
    --label_map_file="projects/fashionpedia/dataset/fashionpedia_label_map.csv" \
    --output_html="out.html" --max_boxes_to_draw=8 --min_score_threshold=0.01 \
    --image_size="640" \
    --image_file_pattern="LOCATION_OF_TAR ToDo" \ # "./fashionveil.tar" or "./fashionpedia.tar"
    --output_file="outputs/spinenet143-ff_test.npy"
```
For `FashionVeil` the images needs to be converted to a `.tar` file and the labels must correspond to the `FashionVeil` labels:

```bash
tar -cvf fashionveil.tar /home/datsplit/model_development/FashionVeil_supercategories/*.png
mkdir -p outputs
mkdir -p preds
python fashionfail/models/convert_amrcnn_labels.py \
    --predictions_path="/home/datsplit/model_development/tpu/models/official/detection/outputs/spinenet143-fv_all.npy" \
    --output_file="./preds/spinenet143-fv_all.npy"
```

Now we can evaluate `amrcnn` on `Fashionpedia-test` and `FashionVeil` completely and per occlusion level.

```bash
    python fashionfail/models/evaluate_new.py \
    --preds_path "/home/datsplit/model_development/fashionfail/src/preds/spinenet143-fv_all.npy" \
    --anns_path "/home/datsplit/model_development/fashionveil_coco.json" \
    --eval_method "COCO" \
    --model_name "amrcnn" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionveil"

python fashionfail/models/evaluate_new.py \
    --preds_path "/home/datsplit/model_development/fashionfail/src/preds/spinenet143-fv_all.npy" \
    --anns_path "/home/datsplit/model_development/fashionveil_coco.json" \
    --eval_method "COCO" \
    --model_name "amrcnn" \
    --iou_type "bbox" \
    --occlusion_anns True \
    --benchmark_dataset "fashionveil"
```


### Inference on FashionFormer [[paper]][paper_fformer] [[code]][code_fformer]

> [!IMPORTANT]
> The following content is adapted from [Velioglu et at.](https://github.com/rizavelioglu/fashionfail):

Create and activate the conda environment:
```bash
conda create -n fformer python==3.8.13
conda activate fformer
```

Install dependencies:
```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
pip install -U openmim
mim install mmdet==2.18.0
mim install mmcv-full==1.3.18
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -U scikit-learn
pip install -U scikit-image
pip install torchmetrics
```

Clone the repository and create a new directory for the model weights:
```bash
cd /change/dir/to/fashionfail/repo/
git clone https://github.com/xushilin1/FashionFormer.git
mkdir FashionFormer/ckpts
```
Download the models manually from [OneDrive][models_fformer] and place them inside the newly created
`FashionFormer/ckpts` folder.

```bash
python src/fashionfail/models/predict_fformer.py \
    --model_path "./FashionFormer/ckpts/fashionformer_swin_b_3x.pth" \
    --config_path  "./FashionFormer/configs/fashionformer/fashionpedia/fashionformer_swin_b_mlvl_feat_6x.py"\
    --out_dir "predictions_fashionformer_swinb/" \
    --image_dir "/home/datsplit/model_development/FashionVeil_supercategories" \
    --dataset_name "fashionveil" \
    --fashionveil_mapping True \
    --score_threshold 0.05

python fashionfail/models/evaluate.py \
    --preds_path "/home/datsplit/model_development/fashionfail/predictions_fashionformer_all/fashionformer_r50_3x-fashionveil.npz" \
    --anns_path "/home/datsplit/model_development/fashionveil_coco.json" \
    --eval_method "COCO" \
    --model_name "fformer" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionveil"


python fashionfail/models/evaluate.py \
    --preds_path "/home/datsplit/model_development/fashionfail/predictions_fashionformer_all/fashionformer_r50_3x-fashionveil.npz" \
    --anns_path "/home/datsplit/model_development/fashionveil_coco.json" \
    --eval_method "COCO" \
    --model_name "fformer" \
    --iou_type "bbox" \
    --occlusion_anns True \
    --benchmark_dataset "fashionveil"

python fashionfail/models/evaluate.py \
    --preds_path "/home/datsplit/model_development/fashionfail/predictions_fashionformer_swinb/fashionformer_swin_b_3x-fashionveil.npz" \
    --anns_path "/home/datsplit/model_development/fashionveil_coco.json" \
    --eval_method "COCO" \
    --model_name "fformer" \
    --iou_type "bbox" \
    --benchmark_dataset "fashionveil"



python src/fashionfail/models/predict_fformer.py \
    --model_path "./FashionFormer/ckpts/fashionformer_r50_3x.pth" \
    --config_path  "./FashionFormer/configs/fashionformer/fashionpedia/fashionformer_r50_mlvl_feat_3x.py"\
    --out_dir "predictions_fashionformer_all/" \
    --image_dir "/home/datsplit/model_development/FashionVeil_supercategories" \
    --dataset_name "fashionveil" \
    --fashionveil_mapping True \
    --score_threshold 0.05
```



> Note: A `score_threshold=0.05` is applied to model predictions. This is because the `fformer` outputs a fixed
> number (100) of predictions for each input due to its Transformer architecture, resulting in many unconfident and
> mainly wrong predictions, which can lead to poor results. Therefore, this thresholding is applied to evaluate the
> model's performance fairly.



[paper_fformer]: https://arxiv.org/abs/2204.04654
[code_fformer]: https://github.com/xushilin1/FashionFormer
[models_fformer]: https://1drv.ms/u/s!Ai4mxaXd6lVBcAWlLG9x3sx8cKY?e=cBZdNy
[paper_amrcnn]: https://arxiv.org/abs/2004.12276
[code_amrcnn]: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/fashionpedia
[amrcnn_spine_ckpt]: https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz
[amrcnn_spine_cfg]: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml
[amrcnn_r50_ckpt]: https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz
[amrcnn_r50_cfg]: https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/r50fpn_amrcnn.yaml
[facere_onnx]: https://huggingface.co/rizavelioglu/fashionfail/resolve/main/facere_base.onnx?download=true




---
<div style="display: flex; justify-content: space-between;">

   [Back](02_training.md)

   [Next: Visualization](04_visualization.md)

</div>