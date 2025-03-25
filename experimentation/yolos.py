# Standard library imports
import logging

# Third-party imports
import lightning as pl
import matplotlib.pyplot as plt
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import YolosForObjectDetection
import config


class Yolosbase(pl.LightningModule):

    def __init__(self, learning_rate, weight_decay, _cats, optimizer_name="adam", momentum=None, beta1=None, beta2=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self._cats = _cats
        self.model = YolosForObjectDetection.from_pretrained(
            "hustvl/yolos-base", num_labels=self._cats, ignore_mismatched_sizes=True, attn_implementation="sdpa")

    def convert_batch_to_target(self, batch):
        targets = []
        for label in batch['labels']:
            target = {
                'boxes': label['boxes'],
                'labels': label['class_labels']
            }
            targets.append(target)
        return targets

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def common_step_val_test(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, batch_size=config.BATCH_SIZE)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=config.BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step_val_test(batch, batch_idx)
        self.log("validation_loss", loss, batch_size=config.BATCH_SIZE)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=config.BATCH_SIZE)

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step_val_test(batch, batch_idx)
        self.log("validation_loss", loss, batch_size=config.BATCH_SIZE)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=config.BATCH_SIZE)

        return loss, loss_dict, outputs

    # From: https://github.com/rizavelioglu/fashionfail/blob/main/src/fashionfail/models/facere.py#L33
    def optimizer_parameters(self):
        if self.optimizer_name == "sgd":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 0.005,
                "momentum": self.momentum if self.momentum else 0.9,
                "weight_decay": self.weight_decay if self.weight_decay else 0.0005,
            }
        elif self.optimizer_name == "adam":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 1e-3,
                "betas": (
                    self.beta1 if self.beta1 else 0.9,
                    self.beta2 if self.beta2 else 0.95,
                ),
                "weight_decay": self.weight_decay if self.weight_decay else 0,
            }
        else:
            raise ValueError(f"Optimizer `{self.optimizer_name}` unknown")
        return parameters

    # From: https://github.com/rizavelioglu/fashionfail/blob/main/src/fashionfail/models/facere.py#L33
    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), **self.optimizer_parameters()
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), **self.optimizer_parameters()
            )
        else:
            raise ValueError(f"Cannot configure optimizer `{
                             self.optimizer_name}`")
        return optimizer
