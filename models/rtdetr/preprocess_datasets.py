# pylint: disable=missing-module-docstring
from typing import Dict, Any

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import lightning as pl
import datasets
from loguru import logger

from utils import fix_channels, rescale_bboxes, xyxy_to_xcycwh


class FashionpediaDataPreprocessor(pl.LightningDataModule):
    """A class for preprocessing the Fashionpedia dataset for object detection."""

    def __init__(self, feature_extractor, batch_size: int = 1, num_workers: int = 28):
        """
        Initializes the DataPreprocessor with the given feature extractor, batch size, and number of workers.

        Args:
            feature_extractor: The feature extractor to use for preprocessing.
            batch_size (int, optional): The batch size for the dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers for the dataloaders. Defaults to 27.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.cats = datasets.load_dataset(
            "detection-datasets/fashionpedia", split="val").features['objects'].feature['category']
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.prepare_data_per_node = False
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    # Credits to https://github.com/valentinafeve/fine_tunning_YOLOS_for_fashion

    def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a batch of data for object detection training and validation.

        Args:
            batch (Dict[str, Any]): A dictionary containing image and object data.

        Returns:
            Dict[str, Any]: A dictionary with transformed image and labels.
        """
        inputs = {}
        image = batch['image']
        image = fix_channels(ToTensor()(image[0]))
        inputs['pixel_values'] = self.feature_extractor(
            [image], return_tensors='pt')['pixel_values']
        labels = []
        bbox = [rescale_bboxes(batch['objects'][i]['bbox'], (batch['width']
                                                             [i], batch['height'][i])) for i in range(len(batch['objects']))]
        bbox = [xyxy_to_xcycwh(torch.Tensor(bbox_i)) for bbox_i in bbox]
        labels.append({
            "boxes": bbox,
            "class_labels": [obj['category'] for obj in batch['objects']],
            "image_id": torch.Tensor([batch['image_id']]).int(),
            "area": [obj['area'] for obj in batch['objects']],
            "iscrowd": torch.Tensor([0 for _ in batch['objects']]).int(),
            "orig_size": torch.Tensor([(batch['width'], batch['height'])]).int(),
            "size": torch.Tensor([inputs['pixel_values'].shape[1:]])[0].int(),
        })
        inputs['labels'] = labels
        return inputs
    # Credits to https://github.com/valentinafeve/fine_tunning_YOLOS_for_fashion

    def collate_fn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collates a batch of data for object detection training and validation.

        Args:
            batch (Dict[str, Any]): A dictionary containing image and object data.

        Returns:
            Dict[str, Any]: A dictionary with collated image and labels.
        """
        collated = {}
        collated["pixel_values"] = self.feature_extractor.pad(
            [item['pixel_values'] for item in batch], return_tensors="pt")['pixel_values']
        collated["labels"] = []
        for item in batch:
            item['labels']['boxes'] = torch.stack(item['labels']['boxes'])[0]
            item['labels']['area'] = torch.Tensor(item['labels']['area'])
            item['labels']['class_labels'] = torch.Tensor(
                item['labels']['class_labels'])[0]
            item['labels']['class_labels'] = item['labels']['class_labels'].type(
                torch.LongTensor)
            collated["labels"].append(item['labels'])
        return collated

    def prepare_data(self):

        pass

    def setup(self, stage):
        logger.info("Loading Fashionpedia dataset...")
        self.train_ds = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
            "train", from_=0, to=75, unit="%", rounding="pct1_dropremainder")).with_transform(self.transform)
        self.val_ds = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
            "train", from_=75, to=100, unit="%", rounding="pct1_dropremainder")).with_transform(self.transform)
        self.test_ds = datasets.load_dataset(
            "detection-datasets/fashionpedia", split=datasets.ReadInstruction("val")).with_transform(self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=10)

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=10)

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
