from setuptools import setup, find_packages
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
fashionfail_path = current_dir / "fashionfail"


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
