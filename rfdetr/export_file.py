from rfdetr import RFDETRLarge

model = RFDETRLarge(
    pretrain_weights="rfdetr_large_results/checkpoint_best_ema.pth", resolution=1120)

model.export(output_dir="onnx-models", infer_dir=None,
             simplify=False,  backbone_only=False)
