import lightning as pl
import torch
from lightning import Trainer
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor
import datasets
from preprocess_datasets import DataPreprocessor


class Yolosbase(pl.LightningModule):

    def __init__(self, lr, weight_decay, train_dataloader, val_dataloader, cats):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.cats = cats
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base",
                                                                 num_labels=self.cats.num_classes, ignore_mismatched_sizes=True)

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

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


if __name__ == "__main__":
    train_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
        "train", from_=0, to=95, unit="%", rounding="pct1_dropremainder"))
    val_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
        "train", from_=95, to=100, unit="%", rounding="pct1_dropremainder"))

    cats = train_dataset.features['objects'].feature['category']
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "hustvl/yolos-base", size=816, longest_edge=864)

    preprocessor = DataPreprocessor(feature_extractor)
    preprocessor.prepare_dataloaders(train_dataset, val_dataset)

    train_dataloader = preprocessor.train_dataloader
    val_dataloader = preprocessor.val_dataloader

    model = Yolosbase(lr=2.5e-5, weight_decay=1e-4, train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader, cats=cats)
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)
    torch.set_float32_matmul_precision('medium')

    trainer.fit(model)
    model.model.push_to_hub("detr-resnet-101-fashionpedia")
