import onnxruntime as ort
import onnx
from rfdetr import RFDETRBase

model = RFDETRBase(
    pretrain_weights="/home/datsplit/FashionVeil/models/rfdetr/rfdetr_b_fashionpedia-divest-only/checkpoint_best_ema.pth", resolution=1120)

model.export(output_dir="exported_model", infer_dir=None,
             simplify=False,  backbone_only=False)


onnx_model_path = "exported_model/inference_model.onnx"
onnx_model = onnx.load(onnx_model_path)

for input_info in onnx_model.graph.input:
    print(f"Input name: {input_info.name}")
    shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    print(f"Input shape: {shape}")
