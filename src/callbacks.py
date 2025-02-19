from lightning.pytorch.callbacks import Callback


class PrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting training!")

    def on_train_end(self, trainer, pl_module):
        print("Finished training!")
