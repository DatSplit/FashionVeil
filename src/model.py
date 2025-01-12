import lightning as pl
import torch
from lightning import Trainer
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor
import datasets
from preprocess_datasets import DataPreprocessor
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html#module-interface
# https://www.youtube.com/watch?v=NjF1ZpRO4Ws&ab_channel=AladdinPersson


class Yolosbase(pl.LightningModule):

    def __init__(self, lr, weight_decay, _cats):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self._cats = _cats
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

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)
        for k, v in loss_dict.items():
            self.log("test_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # Scheduler (optionally))
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer


if __name__ == "__main__":
    train_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
        "train", from_=0, to=95, unit="%", rounding="pct1_dropremainder"))
    val_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction(
        "train", from_=95, to=100, unit="%", rounding="pct1_dropremainder"))
    test_dataset = datasets.load_dataset(
        "detection-datasets/fashionpedia", split="val")

    cats = train_dataset.features['objects'].feature['category']
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "hustvl/yolos-base", size=816, longest_edge=864)

    preprocessor = DataPreprocessor(feature_extractor)
    preprocessor.prepare_dataloaders(train_dataset, val_dataset, test_dataset)

    train_dataloader = preprocessor.train_dataloader
    val_dataloader = preprocessor.val_dataloader
    test_dataloader = preprocessor.test_dataloader

    model = Yolosbase(lr=2.5e-5, weight_decay=1e-4, cats=cats)
    trainer = Trainer(max_epochs=1, accelerator="gpu",
                      devices=1, fast_dev_run=True)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.validate(model, val_dataloader)
    trainer.test(model, test_dataloader)
    # trainer.test(model, test_dataloader)
    # trainer.tune/validate/test
    # overfit batches for sanity check
    # fast_dev_run TEST set-up
    # model.model.push_to_hub("detr-resnet-101-fashionpedia")
