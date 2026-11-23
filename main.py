import argparse
import collections
import random

import numpy as np
import torch
from torch import optim

from data_loader import MedicalDataLoader
from model import build_loss, build_model, dice_coeff, iou_score
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

try:  # wandb is optional: only needed when use_wandb / sweep are enabled
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def build_optimizer(model, cfg):
    """AdamW with no weight decay on biases / norm layers / 1-d params."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    use_fused = bool(cfg.get("fused", False)) and torch.cuda.is_available()
    return optim.AdamW(
        groups,
        lr=cfg["lr"],
        amsgrad=cfg.get("amsgrad", False),
        fused=use_fused,
    )


def build_scheduler(optimizer, config, steps_per_epoch=None):
    sched_cfg = dict(config["lr_scheduler"])
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=config["trainer"]["epochs"],
        steps_per_epoch=steps_per_epoch,
        **sched_cfg,
    )


def main(config):
    logger = config.get_logger("train")
    if config["use_wandb"] and wandb is None:
        raise RuntimeError(
            "use_wandb is true but wandb is not installed (uv sync / pip install wandb)"
        )

    if config["use_wandb"] and config["sweep"]:
        wandb.init(project=config["name"], config=config.config)
        for key, value in wandb.config.items():
            if "." in key:
                # e.g. "optimizer.lr" -> config["optimizer"]["lr"]
                section, param = key.split(".", 1)
                if section in config.config:
                    config.config[section][param] = value
            else:
                config.config[key] = value
    logger.info(f"configuration: {config.config}")

    # regular (map-style) dataset -> the loader can shuffle on its own
    data_loader = MedicalDataLoader(seed=SEED, **config["data_loader"])
    valid_data_loader = data_loader.split_validation()
    logger.info(
        f"train={len(data_loader.dataset)} "
        f"val={len(valid_data_loader.dataset) if valid_data_loader else 0} "
        f"test={len(data_loader.split_test().dataset) if data_loader.split_test() else 0}"
    )

    model = build_model(**config["arch"])
    logger.info(model)
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = build_loss(config["loss"]).to(device)
    metrics = [dice_coeff, iou_score]

    optimizer = build_optimizer(model, config["optimizer"])
    lr_scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(data_loader))

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

    if wandb is not None and wandb.run is not None:
        try:
            wandb.finish(exit_code=0)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config file path (default: ./config.json)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;batch_size"),
        CustomArgs(["--model", "--model_type"], type=str, target="arch;model_type"),
        CustomArgs(["--backbone"], type=str, target="arch;backbone"),
        CustomArgs(["--segformer_name"], type=str, target="arch;segformer_name"),
        CustomArgs(["--data_dir"], type=str, target="data_loader;data_dir"),
    ]
    args.parse_args = lambda: args.parse_known_args()[0]
    config = ConfigParser.from_args(args, options)
    main(config)
