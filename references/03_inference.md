# Inference

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

### Inference on RF-DETR
### Inference on Facere
### Inference on Attribute Mask R-CNN [[paper]][paper_amrcnn] [[code]][code_amrcnn]
### Inference on FashionFormer [[paper]][paper_fformer] [[code]][code_fformer]






















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