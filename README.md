# FashionVeil
### Fashion apparel benchmarking dataset with occlusion level annotations for clothes and accessories

<img src="fashionveil_logo.png" alt="FashionVeil" width="30%">


This repository contains code that describes how to downloads the novel FashionVeil dataset and how to train new models and evaluate existing and new fashion apparel detection models.


### Installation

Create a virtual environment named `fv` (short for FashionVeil):

```bash
python3 -m venv fv
source fv/bin/activate
git clone https://github.com/DatSplit/FashionVeil.git
pip install -e .
```

### Usage
Follow the instructions in `references/` for instructions on [downloads](references/01_dataset_downloads.md), [training](references/02_training.md), [inference and evaluation](references/03_inference_and_evaluation), and [visualization](references/04_visualization.md).