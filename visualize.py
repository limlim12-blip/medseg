"""Plot learning curves and image / ground-truth / prediction grids.

Learning curves read the ``history.json`` written by the Trainer next to each
checkpoint; predictions are made from a checkpoint (the model is rebuilt from
the config stored in it).

    python visualize.py --run saved/models/<name>/<run_id> [--run <other run> ...]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from data_loader import MedicalDataLoader  # noqa: E402
from model import dice_coeff, load_model  # noqa: E402
from utils import prepare_device  # noqa: E402


def plot_learning_curves(run_dirs, out_path):
    panels = [
        ("val_loss", "Validation loss", 1.0),
        ("val_dice_coeff", "Validation Dice (%)", 100.0),
        ("val_iou_score", "Validation IoU (%)", 100.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 5))
    plotted = False
    for run in run_dirs:
        history_file = Path(run) / "history.json"
        if not history_file.exists():
            continue
        history = json.loads(history_file.read_text())
        label = Path(run).parent.name + "/" + Path(run).name
        epochs = [h["epoch"] for h in history]
        for ax, (key, title, scale) in zip(axes, panels):
            if all(key in h for h in history):
                ax.plot(epochs, [h[key] * scale for h in history], label=label, linewidth=2)
                plotted = True
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.6)
    if not plotted:
        plt.close(fig)
        return
    axes[0].legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def visualize_predictions(run_dirs, num_samples, out_path, split="test"):
    device, _ = prepare_device(1)
    models, names, data_loader = [], [], None
    for run in run_dirs:
        ckpt = Path(run) / "model_best.pth"
        if not ckpt.exists():
            continue
        model, config = load_model(ckpt, device)
        model.eval()
        models.append(model)
        names.append(Path(run).parent.name + "/" + Path(run).name)
        if data_loader is None:
            loader_cfg = dict(config["data_loader"], num_workers=0)
            data_loader = MedicalDataLoader(**loader_cfg)
    if not models:
        return

    loader = {"val": data_loader.split_validation(), "test": data_loader.split_test()}[split]
    dataset = (loader or data_loader).dataset
    num_samples = min(num_samples, len(dataset))

    fig, axes = plt.subplots(
        num_samples, 2 + len(models), figsize=(3.5 * (2 + len(models)), 3.5 * num_samples), squeeze=False
    )
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    for i in range(num_samples):
        image, mask = dataset[i]
        shown = np.clip(image.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        axes[i, 0].imshow(shown)
        axes[i, 0].set_title(f"Sample #{i + 1}")
        axes[i, 1].imshow(mask.squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        for j, (model, name) in enumerate(zip(models, names)):
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            dice = dice_coeff(logits.cpu(), mask.unsqueeze(0))
            axes[i, 2 + j].imshow(pred, cmap="gray")
            axes[i, 2 + j].set_title(f"{name}\nDice {dice * 100:.1f}%", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--run", action="append", required=True, help="run dir (saved/models/<name>/<run_id>); repeatable")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out_dir", default="saved/figures")
    args = parser.parse_args()

    plot_learning_curves(args.run, Path(args.out_dir) / "learning_curves.png")
    visualize_predictions(
        args.run, args.num_samples, Path(args.out_dir) / "segmentation_comparison.png", args.split
    )


if __name__ == "__main__":
    main()
