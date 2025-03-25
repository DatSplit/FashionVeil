import torch  # noqa
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import YolosImageProcessor
from aim.pytorch_lightning import AimLogger

from experimentation.yolos import Yolosbase
from preprocess_datasets import FashionpediaDataPreprocessor
import config
from callbacks import PrintingCallback


if __name__ == "__main__":
    feature_extractor = YolosImageProcessor.from_pretrained(
        "hustvl/yolos-base")
    dm = FashionpediaDataPreprocessor(feature_extractor)
    model = Yolosbase(learning_rate=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY, _cats=dm.cats.num_classes)
    aim_logger = AimLogger(
        experiment="test",
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )
    cb_ckpt = ModelCheckpoint(
        dirpath="./saved_models/",
        save_top_k=1,
        monitor="validation_loss",
        filename="test" + "-{epoch:02d}-{validation_loss:.2f}",
    )
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(max_epochs=config.NUM_EPOCHS, accelerator="gpu",
                      devices=1, logger=aim_logger, callbacks=[PrintingCallback(), cb_ckpt])

    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    # trainer.test(model, dm)
