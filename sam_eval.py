"""Zero-shot evaluation of the Segment Anything Model (SAM) on the project's test split.

SAM is promptable and not trained on this dataset, so every image gets a prompt derived from
its ground-truth mask (standard protocol for zero-shot medical evaluation, e.g. MedSAM):

    box    the tight bounding box of the lesion
    point  one foreground click: the mask pixel closest to the lesion centroid
    masks: A rough or low-resolution existing segmentation mask used to refine or guide the output.

The split (same seed / ratios as main.py) and resolution (``data_loader.img_size``) match the
supervised models, so the Dice / IoU numbers are directly comparable with ``test.py``.
Note that box / point prompts use ground-truth information, which the supervised U-Net and
SegFormer do not receive; report SAM with that caveat.

    python sam_eval.py --data_dir data/kvasir-seg
    python sam_eval.py --model_name facebook/sam-vit-large --prompt box --num_vis 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Torch-free helpers
# --------------------------------------------------------------------------- #
def load_pair(image_path, mask_path, img_size):
    """RGB PIL image and boolean mask, resized exactly like dataset.py does."""
    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize(img_size, Image.Resampling.BILINEAR)
    )
    mask = Image.open(mask_path).convert("L").resize(img_size, Image.Resampling.NEAREST)
    return image, np.array(mask, dtype=np.float32) > 127.5


def mask_to_box(mask):
    """Tight [x_min, y_min, x_max, y_max] box (pixel coordinates) around a boolean mask."""
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_to_point(mask):
    """[x, y] of the foreground pixel closest to the centroid (always inside the mask)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return [float(xs[i]), float(ys[i])]


def dice_iou(pred, gt, smooth=1e-6):
    """Dice and IoU of two boolean masks (same formula as model.dice_coeff / iou_score)."""
    pred, gt = pred.astype(np.float64), gt.astype(np.float64)
    inter = (pred * gt).sum()
    dice = (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)
    iou = (inter + smooth) / (pred.sum() + gt.sum() - inter + smooth)
    return float(dice), float(iou)


# --------------------------------------------------------------------------- #
# SAM inference
# --------------------------------------------------------------------------- #
def predict(model, processor, image, mask, prompt, device):
    """Run SAM on one image with a box or point prompt built from ``mask``; return a bool mask."""
    import torch

    if prompt == "box":
        prompt_kwargs = {"input_boxes": [[mask_to_box(mask)]]}
    elif prompt == "point":
        prompt_kwargs = {"input_points": [[mask_to_point(mask)]], "input_labels": [[1]]}
    else:
        raise ValueError(f"Unknown prompt '{prompt}' (use 'box' or 'point')")

    inputs = processor(images=image, return_tensors="pt", **prompt_kwargs).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks[0][0, 0].numpy().astype(bool)  # [H, W] at the input image size


def evaluate(model, processor, pairs, prompts, device, img_size, num_vis=0):
    import torch

    scores = {p: {"dice": [], "iou": [], "seconds": 0.0} for p in prompts}
    n_skipped, vis = 0, []

    for idx, (image_path, mask_path) in enumerate(pairs):
        image, gt = load_pair(image_path, mask_path, img_size)
        if not gt.any():  # no lesion -> no box / point to prompt with
            n_skipped += 1
            continue

        preds = {}
        for prompt in prompts:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            pred = predict(model, processor, image, gt, prompt, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            scores[prompt]["seconds"] += time.time() - start

            dice, iou = dice_iou(pred, gt)
            scores[prompt]["dice"].append(dice)
            scores[prompt]["iou"].append(iou)
            preds[prompt] = (pred, dice)

        if len(vis) < num_vis:
            vis.append((image, gt, preds))
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(pairs)} images")

    results = {}
    for prompt, s in scores.items():
        n = len(s["dice"])
        results[prompt] = {
            "dice": float(np.mean(s["dice"])) if n else float("nan"),
            "dice_std": float(np.std(s["dice"])) if n else float("nan"),
            "iou": float(np.mean(s["iou"])) if n else float("nan"),
            "iou_std": float(np.std(s["iou"])) if n else float("nan"),
            # includes SAM preprocessing (resize / normalise), not just the network forward
            "latency_ms": s["seconds"] / max(n, 1) * 1000.0,
            "fps": n / max(s["seconds"], 1e-9),
        }
    return results, n_skipped, vis


def save_visualization(vis, prompts, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(vis)
    fig, axes = plt.subplots(
        n, 2 + len(prompts), figsize=(3.5 * (2 + len(prompts)), 3.5 * n), squeeze=False
    )
    for i, (image, gt, preds) in enumerate(vis):
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Sample #{i + 1}")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground truth")
        for j, prompt in enumerate(prompts):
            pred, dice = preds[prompt]
            axes[i, 2 + j].imshow(pred, cmap="gray")
            axes[i, 2 + j].set_title(
                f"SAM ({prompt}) Dice {dice * 100:.1f}%", fontsize=9
            )
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot SAM evaluation (Dice / IoU)"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="read data_dir / img_size / splits from here",
    )
    parser.add_argument(
        "--data_dir", default=None, help="override data_loader.data_dir"
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--model_name",
        default="facebook/sam-vit-base",
        help="Hugging Face SAM checkpoint: facebook/sam-vit-base | -large | -huge",
    )
    parser.add_argument(
        "--prompt", nargs="+", default=["box", "point"], choices=["box", "point"]
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="evaluate only the first N images (quick check)",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=4,
        help="number of qualitative examples to save (0 = none)",
    )
    parser.add_argument("--out", default="saved/sam_zero_shot_metrics.json")
    parser.add_argument("--fig", default="saved/figures/sam_zero_shot.png")
    args = parser.parse_args()

    try:
        from transformers import SamModel, SamProcessor
    except ImportError as e:
        raise SystemExit(
            "SAM needs 'transformers': pip install transformers (or uv sync --extra transformers)"
        ) from e

    from data_loader.dataset import ensure_dataset, find_image_mask_pairs, split_indices
    from utils import prepare_device, read_json

    cfg = read_json(args.config)["data_loader"]
    data_dir = args.data_dir or cfg["data_dir"]
    img_size = cfg["img_size"]
    img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

    ensure_dataset(data_dir, cfg.get("download", True))
    image_paths, mask_paths = find_image_mask_pairs(data_dir)
    if not image_paths:
        raise SystemExit(f"No image/mask pairs found in '{data_dir}'.")
    # identical split to main.py / test.py (seed 123)
    _, val_idx, test_idx = split_indices(
        len(image_paths), cfg["validation_split"], cfg["test_split"], seed=123
    )
    idx = test_idx if args.split == "test" else val_idx
    if args.max_images:
        idx = idx[: args.max_images]
    pairs = [(image_paths[i], mask_paths[i]) for i in idx]
    if not pairs:
        raise SystemExit(
            f"The '{args.split}' split is empty; adjust validation_split / test_split."
        )

    device, _ = prepare_device(1)
    print(f"Loading {args.model_name} on {device} ...")
    processor = SamProcessor.from_pretrained(args.model_name)
    model = SamModel.from_pretrained(args.model_name).to(device).eval()

    print(f"Evaluating {len(pairs)} {args.split} images, prompts: {args.prompt}")
    results, n_skipped, vis = evaluate(
        model, processor, pairs, args.prompt, device, img_size, args.num_vis
    )

    report = {
        "model": args.model_name,
        "split": args.split,
        "img_size": list(img_size),
        "n_images": len(pairs) - n_skipped,
        "n_skipped_empty_mask": n_skipped,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=4))
    print(json.dumps(report, indent=4))

    if vis:
        save_visualization(vis, args.prompt, args.fig)
        print(f"Saved qualitative examples to {args.fig}")


if __name__ == "__main__":
    main()
