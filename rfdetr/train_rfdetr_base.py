import pandas as pd
import matplotlib.pyplot as plt
from rfdetr import RFDETRBase
import os
model = RFDETRBase(resolution=1120)
dataset_dir = os.path.expanduser("~/.cache/rfdetr_fashionpedia/")
history = []


def callback2(data):
    history.append(data)


model.callbacks["on_fit_epoch_end"].append(callback2)
model.train(dataset_dir=dataset_dir,
            epochs=6, batch_size=4, grad_accum_steps=4, lr=1e-4)
model.export()


df = pd.DataFrame(history)

plt.figure(figsize=(12, 8))

plt.plot(
    df['epoch'],
    df['train_loss'],
    label='Training Loss',
    marker='o',
    linestyle='-'
)

plt.plot(
    df['epoch'],
    df['test_loss'],
    label='Validation Loss',
    marker='o',
    linestyle='--'
)

plt.title('Train/Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
