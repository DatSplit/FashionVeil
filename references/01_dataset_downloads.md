# Dataset downloads

Three datasets are used for training, evaluation and inferencing of the models: Fashionpedia, Fashionpedia-divest, and FashionVeil.

### Fashionpedia for RF-DETR

RF-DETR requires a specific dataset structure, therefore separate versions of Fashionpedia and FashionVeil are available for training and evaluation.

Fashionpedia for RF-DETR is available on request to `n.d.teunissen@students.uu.nl`. The following bash commands can be run to download and extract the dataset given the provided `PRESIGNED_URL`:

```bash
curl -L "PRESIGNED_URL" -o fashionpedia_rfdetr.zip
unzip -o fashionpedia_rfdetr.zip
```

Fashionpedia-divest (Not available yet - around mid July)

### FashionVeil

`FashionVeil` is available on request to `n.d.teunissen@students.uu.nl`. The following bash commands can be run to download and extract the dataset given the provided `PRESIGNED_URL`:

```bash
curl -L "PRESIGNED_URL" -o CHOOSE_A_NAME.zip
unzip CHOOSE_A_NAME
```

There are three `FashionVeil` downloads available to reproduce the fine-tuning and evaluation results:
 1. `FashionVeil_all`, this directory contains all images and two annotation files:
    1. `FashionVeil_all.json` containing annotations for all classes.
    2. `FashionVeil_supercategories.json` only contains the categories also present in `Fashionpedia`
 2. `FashionVeil_train`, this directory contains `FashionVeil_train_all.json` and `FashionVeil_validation_all.json` for all categories. Two sub directories `train` and `valid` with annotation categories only present in `Fashionpedia` and the corresponding images.
 3. `FashionVeil_test`, this directory contains the test-set for all categories `FashionVeil_test_all.json` and the categories only present in `Fashionpedia`, `fashionveil_test.json`.
 

These datasets can now be used to train and/or evaluate existing and new fashion apparel detection models!

---
<div style="display: flex; justify-content: space-between;">


   [Next: Training](02_training.md)

</div>
