from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from huggingface_hub import hf_hub_download
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Get the current directory
current_dir = os.getcwd()

# Path for the TensorRT engine
engine_path = os.path.join(current_dir, "facere_base.trt")
# Convert ONNX to TensorRT
def build_engine(onnx_path, engine_path, precision="fp16"):
    """
    Build TensorRT engine from ONNX file
    
    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path where TensorRT engine will be saved
        precision: Precision mode ('fp32', 'fp16', or 'int8')
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set memory requirements
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # You would need to set up a calibrator here for INT8
    
    # Build and save engine
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    return engine
# Convert the model
build_engine("facere_base.onnx", engine_path, precision="fp16")



