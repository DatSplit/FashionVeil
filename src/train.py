import lightning as pl
import torch
from lightning import Trainer
from transformers import AutoFeatureExtractor, YolosForObjectDetection
from preprocess_datasets import FashionpediaDataPreprocessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import logging
from model import Yolosbase
from preprocess_datasets import FashionpediaDataPreprocessor
import config
from callbacks import PrintingCallback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from aim.pytorch_lightning import AimLogger

if __name__ == "__main__":
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "hustvl/yolos-base", size=816, longest_edge=864)
    dm = FashionpediaDataPreprocessor(feature_extractor)
    model = Yolosbase(lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY, _cats=dm.cats.num_classes)
    aim_logger = AimLogger(
        experiment="test",
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )
    cb_ckpt = ModelCheckpoint(
        dirpath="./saved_models/",
        save_top_k=1,
        monitor="val_loss_sum",
        filename="test" + "-{epoch:02d}-{validation_loss:.2f}",
    )

    trainer = Trainer(max_epochs=config.NUM_EPOCHS, accelerator="gpu",
                      devices=1, logger=aim_logger, callbacks=[PrintingCallback(), cb_ckpt])
    torch.set_float32_matmul_precision('medium')
    # trainer.fit(model, dm)
    # trainer.validate(model, dm)
    trainer.test(model, dm)
