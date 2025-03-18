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
                                       weight_decay=config.WEIGHT_DECAY, _cats=dm.cats.num_classes)
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
    )

    trainer = Trainer(max_epochs=config.NUM_EPOCHS, accelerator="gpu",
                      # accumulate_grad_batches=2
                      devices=1, logger=aim_logger, callbacks=[PrintingCallback(), cb_ckpt], precision="16-mixed")
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    # trainer.test(model, dm)
