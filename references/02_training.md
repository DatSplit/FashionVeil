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

For a more detailed guide on exporting trained models to the `ONNX` format see: https://github.com/DatSplit/rf-detr_export_documentation/blob/feature/export-docs/rfdetr/docs/export.md

