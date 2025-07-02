# Dataset curation

`dataset_preprocessing/curate_xanylabeling_annotations`

The annotated images are exported from X-Any-labeling as a folder containing images and corresponding .json files for each individual image.

For the `FashionVeil` datasets with only the `Fashionpedia` categories, `curate_fashionpedia_categories.py` can be run first.

`convert_to_coco_custom.py`: the individual files are merged into a single custom COCO annotation file, where the occlusion level is the additional field.

`train_val_test_split.py`: was run once to filer all images into `train, val, and test` splits.

`filter_coco_annotations.py`: was run to split the complete `FashionVeil` dataset into the same splits as the `FashionVeil` supercategories dataset.

`clean_unused_images.py`: was run to remove all images in the directory that are not in the corresponding annotation file. 

---
<div style="display: flex; justify-content: space-between;">


   [Next: Dataset downloads](01_dataset_downloads.md)

</div>