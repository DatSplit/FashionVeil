from datetime import datetime

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import RTDetrImageProcessor
from aim.pytorch_lightning import AimLogger
from loguru import logger
from lightning.pytorch import seed_everything

from rtdetr import rtdetr
from preprocess_datasets import FashionpediaDataPreprocessor
import config
from callbacks import PrintingCallback


if __name__ == "__main__":
    # Initialize the feature extractor, data preprocessor, model, logger, and callbacks
    logger.info(f"Checkpoint path: {config.CKPT_PATH}")
    seed_everything(config.SEED, workers=True)
    feature_extractor = RTDetrImageProcessor.from_pretrained(
        f"PekingU/{config.BACKBONE}")
    dm = FashionpediaDataPreprocessor(feature_extractor)
    model = rtdetr(learning_rate=config.LEARNING_RATE,
                   weight_decay=config.WEIGHT_DECAY, optimizer_name="adam", backbone=config.BACKBONE, _cats=dm.cats.num_classes)

    aim_logger = AimLogger(
        experiment=config.EXPERIMENT_NAME,
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )
    cb_ckpt = ModelCheckpoint(
        dirpath="./saved_models/",
        save_top_k=1,
        monitor="validation_loss",
        filename=f"{config.EXPERIMENT_NAME}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{{epoch:02d}}-{{validation_loss:.2f}}",
        save_last=True,
    )
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(max_epochs=config.NUM_EPOCHS, accelerator="gpu",
                      devices=1, logger=aim_logger, callbacks=[PrintingCallback(), cb_ckpt], accumulate_grad_batches=config.BATCH_SIZE, enable_checkpointing=True)

    trainer.fit(
        model, dm, ckpt_path=config.CKPT_PATH)
    trainer.validate(model, dm)
    trainer.test(model, dm)
