import json
import re
from loguru import logger


def get_prefixes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    prefixes = set()
    for img in data['images']:
        match = re.match(r'(p\d+_)', img['file_name'])
        if match:
            prefixes.add(match.group(1))
    return prefixes


splits = {
    'train': 'annotations/fashionveil_train.json',
    'val': 'annotations/fashionveil_val.json',
    'test': 'annotations/fashionveil_test.json',
}

prefix_sets = {split: get_prefixes(path) for split, path in splits.items()}
logger.info(prefix_sets)
for split1 in splits:
    for split2 in splits:
        if split1 >= split2:
            continue
        overlap = prefix_sets[split1] & prefix_sets[split2]
        print(
            f"Overlap between {split1} and {split2}: {overlap if overlap else 'None'}")
