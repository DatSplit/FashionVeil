from setuptools import setup, find_packages
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
fashionfail_path = current_dir / "fashionfail"
print(f"FashionFail path: {fashionfail_path}")
setup(
    name="FashionVeil",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.7.1",
        "lightning==2.5.2",
        "transformers==4.53.0",
        "aim>=3.29.1",
        "loguru>=0.7.3",
        "torchvision==0.22.1",
        "datasets==3.6.0",
        "matplotlib==3.10.3",
        "scipy==1.16.0",
        "rfdetr==1.1.0",
        f"fashionfail @ file://{fashionfail_path}",
        "onnxruntime-gpu==1.22.0",
        "pycocotools==2.0.8",
    ]
)

# "pandas>=2.2.1",
# "requests>=2.31.0",
# "huggingface-hub>=0.21.4",
# "loguru>=0.7.3",
# "numpy",
# "PyYAML",
# "h5py",
# "IPython",
# "torch>=2.7.1",
# "torchaudio>=2.1.1",
# "torchvision>=0.17.1",
# "torchmetrics>=1.2.1",
# "pytorch-lightning>=1.8.6",
# "lightning",
# "transformers",
# "datasets>=2.16.1",
# "scikit-learn>=1.4.1",
# "scikit-image",
# "scipy",
# "Pillow>=10.2.0",
# "opencv-python",
# "matplotlib>=3.8.3",
# "seaborn",
# "streamlit",
# "pycocotools>=2.0.7",
# "supervision==0.6.0",
# "tqdm>=4.66.2",
# "aim>=3.29.1",
# "onnx==1.14.1",
# "onnxruntime-gpu==1.15.0",
# "pytest>=7.2.0",
# "pre-commit>=3.3.2",
# "pip-tools>=6.12.2",
# "tensorflow"
