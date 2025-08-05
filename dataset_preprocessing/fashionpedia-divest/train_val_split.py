from fashionfail.data.make_fashionpedia import split_train_val_and_export
from pathlib import Path
split_train_val_and_export(
    ann_file="fashionpedia_divest_all_classes_annotations.json",
    target_dir=Path(
        "splits"),
    test_size=0.2,
    random_seed=42,
)
