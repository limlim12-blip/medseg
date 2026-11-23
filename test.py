import argparse
import json
import time
from pathlib import Path

import torch

from data_loader import MedicalDataLoader
from model import build_loss, dice_coeff, iou_score, load_model
from utils import prepare_device


def evaluate(model, loader, criterion, device):
    """Loss, Dice, IoU (all averaged per batch) plus latency / throughput."""
    model.eval()
    n_images = len(loader.dataset)
    n_batches = max(1, len(loader))
    loss_sum = dice_sum = iou_sum = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss_sum += criterion(logits, target).item()
            dice_sum += dice_coeff(logits, target)
            iou_sum += iou_score(logits, target)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return {
        "loss": loss_sum / n_batches,
        "dice": dice_sum / n_batches,
        "iou": iou_sum / n_batches,
        "latency_ms": elapsed / max(1, n_images) * 1000.0,
        "fps": n_images / max(elapsed, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the test split"
    )
    parser.add_argument(
        "-r", "--resume", required=True, help="path to checkpoint (.pth)"
    )
    parser.add_argument(
        "-d", "--device", default=None, help="cuda device index (default: all)"
    )
    parser.add_argument(
        "--data_dir", default=None, help="override data_dir from the checkpoint config"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--out",
        default=None,
        help="where to write metrics json (default: next to checkpoint)",
    )
    args = parser.parse_args()

    device, _ = prepare_device(1)
    model, config = load_model(args.resume, device)

    loader_cfg = dict(config["data_loader"])
    if args.data_dir:
        loader_cfg["data_dir"] = args.data_dir
    loader_cfg["num_workers"] = 0
    data_loader = MedicalDataLoader(**loader_cfg)
    loader = {
        "train": data_loader,
        "val": data_loader.split_validation(),
        "test": data_loader.split_test(),
    }[args.split]
    if loader is None:
        raise SystemExit(
            f"The '{args.split}' split is empty; adjust validation_split / test_split in the config."
        )

    criterion = build_loss(config["loss"]).to(device)
    results = evaluate(model, loader, criterion, device)
    results["params_M"] = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )
    results["split"] = args.split

    out = (
        Path(args.out)
        if args.out
        else Path(args.resume).parent / f"{args.split}_metrics.json"
    )
    out.write_text(json.dumps(results, indent=4))
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
