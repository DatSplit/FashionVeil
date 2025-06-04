import pandas as pd
import matplotlib.pyplot as plt
from rfdetr import RFDETRLarge
import os
model = RFDETRLarge(resolution=1120)
dataset_dir = os.path.expanduser("~/.cache/rfdetr_fashionpedia/")
fine_tune_dataset_dir = "/home/datsplit/wearables_detection_airport_security/FashionVeil/"
history = []


def callback2(data):
    history.append(data)


model.callbacks["on_fit_epoch_end"].append(callback2)
model.train(dataset_dir=fine_tune_dataset_dir,
            epochs=100, batch_size=1, grad_accum_steps=16, early_stopping=True, tensorboard=True, output_dir="rfdetr_large_results", checkpoint_interval=1, resume="./rfdetr_large_results/checkpoint_best_ema.pth")
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
