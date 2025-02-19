import lightning as pl
import torch
from lightning import Trainer
from transformers import AutoFeatureExtractor, YolosForObjectDetection
from preprocess_datasets import FashionpediaDataPreprocessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import logging
from pytorch_lightning.callbacks import EarlyStopping


logging.basicConfig(level=logging.INFO)


class Yolosbase(pl.LightningModule):

    def __init__(self, learning_rate, weight_decay, _cats):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._cats = _cats
        # self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base",
        # num_labels=self._cats, ignore_mismatched_sizes=True)
        self.model = YolosForObjectDetection.from_pretrained(
            "DatSplit/yolos-base-fashionpedia", num_labels=self._cats, ignore_mismatched_sizes=True)
        self.map_metric = MeanAveragePrecision(
            class_metrics=True, box_format="cxcywh")

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

        preds = [
            {
                "scores": torch.max(logit, dim=1)[0],
                "boxes": pred_box,
                "labels": torch.argmax(logit, dim=1)
            }
            for logit, pred_box in zip(outputs.logits, outputs.pred_boxes)
        ]

        targets = self.convert_batch_to_target(batch)
        self.map_metric.update(preds, targets)

        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step_val_test(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def on_validation_epoch_end(self):
        map_score = self.map_metric.compute()
        logging.info(f"map_score: {map_score}")
        fig_, ax_ = self.map_metric.plot()
        plt.show()
        self.map_metric.reset()

    def test_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step_val_test(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss, loss_dict, outputs

    def on_test_epoch_end(self):
        map_score = self.map_metric.compute()
        fig_, ax_ = self.map_metric.plot()
        plt.show()
        logging.info(f"map_score: {map_score}")

        self.map_metric.reset()

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
