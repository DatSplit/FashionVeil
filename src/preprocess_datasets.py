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

    def __init__(self, feature_extractor, batch_size: int = 2, num_workers: int = 28):
        """
        Initializes the DataPreprocessor with the given feature extractor, batch size, and number of workers.

        Args:
            feature_extractor: The feature extractor to use for preprocessing.
            batch_size (int, optional): The batch size for the dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers for the dataloaders. Defaults to 27.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.batch_size: int = 1
        self.num_workers: int = 28
        self.cats = datasets.load_dataset(
            "detection-datasets/fashionpedia", split="val").features['objects'].feature['category']
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.prepare_data_per_node = False
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a batch of data for object detection training and validation.
        HuggingFace datasets automatically batches data before transformation.

        Args:
            batch (Dict[str, Any]): A dictionary containing batched image and object data.

        Returns:
            Dict[str, Any]: A dictionary with transformed pixel values and labels.
        """
        # Process the entire batch at once - convert it to expected format
        pixel_values_list = []
        labels_list = []

        for image_id in range(len(batch['image'])):
            # Process image
            image = batch['image'][image_id]
            image = fix_channels(ToTensor()(image))
            pixel_values = self.feature_extractor(
                [image], return_tensors='pt')['pixel_values']

            # Process bounding boxes
            bbox = [rescale_bboxes(batch['objects'][image_id]['bbox'],
                                   (batch['width'][image_id], batch['height'][image_id]))]
            bbox = [xyxy_to_xcycwh(torch.Tensor(bbox_i)) for bbox_i in bbox]
            class_labels = batch['objects'][image_id]['category']
            # Create label for this image
            label = {
                "boxes": bbox[0] if bbox and len(bbox[0]) > 0 else torch.zeros((0, 4)),
                "class_labels": torch.LongTensor(class_labels),
                "image_id": torch.tensor([batch['image_id'][image_id]]).int(),
                "area": torch.Tensor(batch['objects'][image_id]['area']),
                "iscrowd": torch.zeros(len(class_labels)).int(),
                "orig_size": torch.tensor([(batch['width'][image_id], batch['height'][image_id])]).int(),
                "size": torch.tensor([pixel_values.shape[1:]]).int()[0],
            }

            pixel_values_list.append(pixel_values)
            labels_list.append(label)

        # Return properly formatted batch
        return {
            "pixel_values": pixel_values_list,
            "labels": labels_list
        }

    def collate_fn(self, batch):
        """
        Collates batches for the dataloader.

        Args:
            batch: A list of dictionaries from the transform method

        Returns:
            Dict: A dictionary with batched pixel values and labels
        """
        # Flatten the pixel values and labels from all batches
        pixel_values = []
        labels = []

        for item in batch:
            if isinstance(item["pixel_values"], list):
                for i, pv in enumerate(item["pixel_values"]):
                    # Ensure each pixel_value is squeezed to remove batch dimension
                    # Remove batch dimension
                    pixel_values.append(pv.squeeze(0))
                    labels.append(item["labels"][i])
            else:
                # Ensure each pixel_value is squeezed to remove batch dimension
                pixel_values.append(item["pixel_values"].squeeze(0))
                labels.append(item["labels"])

        # Pad the pixel values to the same size
        encoding = self.feature_extractor.pad(
            pixel_values, return_tensors="pt")

        # Create the final batch dictionary
        collated_batch = {
            'pixel_values': encoding['pixel_values'],
            'labels': labels
        }
        return collated_batch

    def prepare_data(self):
        # Download or prepare your dataset here
        pass

    def setup(self, stage):
        print("Loading Fashionpedia dataset...")
        self.train_ds = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
            "train", from_=0, to=95, unit="%", rounding="pct1_dropremainder")).with_transform(self.transform)
        self.val_ds = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
            "train", from_=95, to=100, unit="%", rounding="pct1_dropremainder")).with_transform(self.transform)
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
