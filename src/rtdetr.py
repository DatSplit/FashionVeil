from typing import Union, Tuple

from transformers.models.rt_detr.modeling_rt_detr import RTDetrObjectDetectionOutput
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrConfig
import lightning as pl
import torch


def initialize_model(backbone, cats):
    if backbone == "default":
        model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd", num_labels=cats, ignore_mismatched_sizes=True)
    if backbone == "rtdetr_v2_r101vd":
        model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_v2_r101vd", num_labels=cats, ignore_mismatched_sizes=True)
        return model


class rtdetr(pl.LightningModule):
    """ RT-DETR model for the Fashionpedia dataset"""

    def __init__(self, learning_rate: float, weight_decay: float, _cats: int, optimizer_name: str, backbone: str, momentum=None, beta1=None, beta2=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self._cats = _cats
        self.model = initialize_model(backbone, _cats)

    def forward(self, pixel_values: torch.Tensor) -> Union[RTDetrObjectDetectionOutput, Tuple[torch.FloatTensor, ...]]:
        """Forward pass of the RT-DETR model.

        This method passes the input images through the model and returns detection outputs.
        When called during inference (without labels), this returns bounding boxes, class scores,
        and other detection outputs.

        Args:
            pixel_values (torch.Tensor): The pixel values of the input images,
                with shape (batch_size, num_channels, height, width).

        Returns:
            Union[RTDetrObjectDetectionOutput, Tuple[torch.FloatTensor, ...]]:
                Either an RTDetrObjectDetectionOutput object containing loss, scores,
                and bounding boxes (when used with labels) or a tuple of tensors containing
                prediction scores and boxes (during inference).
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    def common_step(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        " Common step for training, validation, and testing. "
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values,
                             labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch: dict) -> torch.Tensor:
        " Training step for the RT-DETR model. "
        loss, loss_dict = self.common_step(batch)
        self.log("training_loss", loss, batch_size=1)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=1)

        return loss

    def validation_step(self, batch: dict) -> torch.Tensor:
        " Validation step for the RT-DETR model. "
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss, batch_size=1)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=1)

        return loss

    def test_step(self, batch: dict) -> Tuple[torch.Tensor, dict, RTDetrObjectDetectionOutput]:
        " Test step for the  RT-DETR model. "
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss, batch_size=1)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=1)
        return loss, loss_dict

    # From: https://github.com/rizavelioglu/fashionfail/blob/main/src/fashionfail/models/facere.py#L33
    def optimizer_parameters(self) -> dict:
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
    def configure_optimizers(self) -> dict:
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
