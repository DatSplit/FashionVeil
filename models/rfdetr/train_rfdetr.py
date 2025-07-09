import pandas as pd
import matplotlib.pyplot as plt
from rfdetr import RFDETRBase, RFDETRLarge
import os
import random
import numpy as np
import torch
import argparse


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    parser = argparse.ArgumentParser(description='Train RFDETR models')
    parser.add_argument('--model_type', type=str, choices=['base', 'large'], required=True,
                        help='Model type: "base" for RFDETR-B or "large" for RFDETR-L')
    parser.add_argument('--dataset_dir', type=str,
                        default=os.path.expanduser(
                            "~/.cache/rfdetr_fashionpedia/"),
                        help='Path to dataset directory, structured according to the RFDETR dataset format')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 2 for base, 1 for large)')
    parser.add_argument('--grad_accum_steps', type=int, default=16,
                        help='Gradient accumulation steps (default: 8 for base, 16 for large)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resolution', type=int, default=1120,
                        help='Image resolution')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for training results (default: rfdetr_{model_type}_results)')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Checkpoint saving interval')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    if args.resume is not None:
        args.resume = os.path.expanduser(args.resume)

    if args.output_dir is None:
        args.output_dir = f"rfdetr_{args.model_type}_results"

    set_seed(42)

    if args.model_type == 'base':
        model = RFDETRBase(resolution=args.resolution)
    else:
        model = RFDETRLarge(resolution=args.resolution)

    history = []

    def callback(data):
        history.append(data)

    model.callbacks["on_fit_epoch_end"].append(callback)

    model.train(dataset_dir=args.dataset_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                early_stopping=True,
                tensorboard=True,
                output_dir=args.output_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume=args.resume)
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

    plt.title(
        f'Train/Validation Loss over epochs - RFDETR-{args.model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
