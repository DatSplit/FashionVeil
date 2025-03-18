# pylint: disable=missing-module-docstring
from typing import Dict, Any, Optional

from transformers import AutoFeatureExtractor
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

from utils import fix_channels, rescale_bboxes, xyxy_to_xcycwh


class DataPreprocessor:
    def __init__(self, feature_extractor: AutoFeatureExtractor, batch_size: int = 1, num_workers: int = 27):
        """
        Initializes the DataPreprocessor with the given feature extractor, batch size, and number of workers.

        Args:
            feature_extractor (AutoFeatureExtractor): The feature extractor to use for preprocessing.
            batch_size (int, optional): The batch size for the dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers for the dataloaders. Defaults to 27.
        """
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None

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

    def prepare_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset) -> None:
        """
        Prepares the dataloaders for training and validation datasets.

        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
        """
        prepared_train = train_dataset.with_transform(self.transform)
        prepared_val = val_dataset.with_transform(self.transform)
        prepared_test = test_dataset.with_transform(self.transform)
        self.train_dataloader = DataLoader(
            prepared_train, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(
            prepared_val, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_dataloader = DataLoader(
            prepared_test, collate_fn=self.collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)
