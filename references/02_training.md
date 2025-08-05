# Training

Using the downloaded datasets we can train RT-DETR and RF-DETR models.

### RT-DETR

To train `RT-DETR` move to the models directory and run `RT-DETR`:

```bash
cd models/rtdetr
python3 train_rtdetr.py
```

Optionally, the configuration of hyperparameters can be changed in `config.py`. 

After training the model must be exported to the `ONNX` format for inference on the benchmark datasets.

```bash
python3 exports/export_to_onnx.py \
    --checkpoint  "PATH_TO_AND_NAME_OF_THE_TRAINED_RT_DETR_MODEL.ckpt" \
    --output "NAME_OF_ONNX_MODEL.onnx" \
    --pretrained_model "PekingU/rtdetr_v2_r101vd" OR "PekingU/rtdetr_v2_r50vd" \
    --num_classes 46
```
`pretrained_model` should be the same as `BACKBONE` in `config.py`.
The `ONNX` model is saved in the folder `models/onnx_models`.


### RF-DETR

To train `RF-DETR-B` and `RF-DETR-L` move to the `models/rfdetr` directory and run the following command:

`dataset-dir` is the only required argument, it should point to one of the `RF-DETR` formatted COCO datasets of `Fashionpedia` or `FashionVeil`.

```bash
cd models/rfdetr
python3 train_rfdetr.py \
    --dataset_dir PATH_TO_RFDETR_FASHIONPEDIA_DATASET \
    --model_type "base" OR "large" \
    --output_dir PATH_FOR_MODEL_CHECKPOINTS \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --resolution 1120
```

After training the model is automatically exported to `ONNX` in the `output` folder.

For a more detailed guide on exporting trained models to the `ONNX` format and `TensorRT` see the section below:

After training a custom RF-DETR model it is often desirable to export the model.
RF-DETR supports exporting models to both ONNX and TensorRT formats.
Exporting models to ONNX enables interoperability with various inference frameworks and can improve deployment efficiency.
Exporting to TensorRT typically reduces inference latency and model size.

## ONNX export

> [!IMPORTANT]
> Starting with RF-DETR 1.2.0, you'll have to run `pip install rfdetr[onnxexport]` before exporting model weights to ONNX format.  
To export your model, simply initialize it and call the `.export()` method. There are several optional arguments that you can pass to the `.export()` method. 

*   `output_dir`: The directory where the ONNX model should be saved.
*   `infer_dir`: A directory where a single sample image exists.
*   `simplify`: A boolean indicating whether you want to simplify the ONNX model. This improves inference speed and reduces model complexity and size.
*   `backbone_only`: A boolean indicating whether you want to export the backbone only. Setting this boolean to true renders the model unable to perform object detection.
*   `resolution`: The resolution on which the model was trained on.

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>, resolution=1120)

model.export(output_dir="onnx-models", infer_dir=None, simplify=True,  backbone_only=False)
```

## TensorRT conversion

> [!IMPORTANT]
> TensorRT conversion must be done on the same device where you want to run inference. 
The ONNX model can be exported to TensorRT for faster inference and reduced model size.
First download and install TensorRT>=8.6.1 from [TensorRT](https://developer.nvidia.com/tensorrt/download), make sure that the TensorRT is compatible with your OS (`lsb_release -a`) and CUDA (`nvcc --version`) version.

To export your ONNX model to TensorRT, initialize and call the `trtexec()` method with the path to your ONNX model and three arguments:

1. `verbose` [tensorRT_docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.)
2. If you want to use nsight-systems profiling install it from [nsight-systems_installation](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html). Documentation is available at [nsight-systems_docs](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli). This provides you with detailed information about TensorRT execution on the GPU.
3. Setting `dry_run` to true enables you to print the command that would be executed.

Run the code below to convert your ONNX model to TensorRT. Change `onnx_model_path`.

```python
from rfdetr.deploy.export import trtexec
import argparse

args = argparse.Namespace()
args.verbose = True
args.profile = False
args.dry_run = False
args.wandb = False # This is required for rf-detr 1.0.0 and 1.1.0! (Will be dropped from 1.2.0 onwards)
onnx_model_path = "your_onnx_model.onnx"

trtexec(onnx_model_path, args)
```
This script will create a file named `your_onnx_model.engine`.

The exported .engine model can be used to perform real-time inference.

---
<div style="display: flex; justify-content: space-between;">

   [Back](01_dataset_downloads.md)

   [Next: Inference and evaluation](03_inference_and_evaluation.md)

</div>

<!-- 
cd models/rfdetr
python3 train_rfdetr.py \
    --dataset_dir "~/.cache/rfdetr_fashionpedia-divest" \
    --model_type "base" \
    --output_dir "rfdetr_b_fashionpedia-divest" \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --resolution 1120 -->