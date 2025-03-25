import torch  # noqa
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import DeformableDetrImageProcessor
from aim.pytorch_lightning import AimLogger

from src.deformable_detr import DeformableDetrFashionpedia
from preprocess_datasets import FashionpediaDataPreprocessor
import config
from callbacks import PrintingCallback


if __name__ == "__main__":
    feature_extractor = DeformableDetrImageProcessor()
    dm = FashionpediaDataPreprocessor(feature_extractor)
    model = DeformableDetrFashionpedia(learning_rate=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY, optimizer_name=config.OPTIMIZER, _cats=dm.cats.num_classes)
    aim_logger = AimLogger(
        experiment=config.EXPERIMENT_NAME,
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )
    cb_ckpt = ModelCheckpoint(
        dirpath="./saved_models/",
        save_top_k=1,
        monitor="validation_loss",
        filename=config.EXPERIMENT_NAME + "-{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )

    trainer = Trainer(max_epochs=config.NUM_EPOCHS, accelerator="gpu",
                      # accumulate_grad_batches=2
                      # Add gradient clipping to prevent NaN values
                      devices=1, logger=aim_logger, callbacks=[PrintingCallback(), cb_ckpt], accumulate_grad_batches=16,  precision="16-mixed",      gradient_clip_val=0.5,
                      gradient_clip_algorithm="norm")
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
