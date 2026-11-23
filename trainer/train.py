import torch

from base import BaseTrainer
from utils import MetricTracker, write_json

try:  # wandb is optional
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class Trainer(BaseTrainer):
    """
    Trainer class. One epoch = one full pass over the (map-style) training set.
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader,
        lr_scheduler,
        do_validation=True,
    ):
        super().__init__(
            model,
            criterion,
            metric_ftns,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader,
            lr_scheduler,
        )
        self.do_validation = do_validation and valid_data_loader is not None
        self.use_wandb = bool(config["use_wandb"]) and wandb is not None
        self.use_amp = (
            bool(config["trainer"].get("amp", False)) and device.type == "cuda"
        )

        self.global_train_step = 0
        self.global_val_step = 0
        wandb_id = None
        if self.resume_ckpt is not None:
            self.global_train_step = self.resume_ckpt.get("global_train_step", 0)
            self.global_val_step = self.resume_ckpt.get("global_val_step", 0)
            wandb_id = self.resume_ckpt.get("wandb_id")

        metric_names = [m.__name__ for m in self.metric_ftns]
        self.train_metrics = MetricTracker(
            "loss",
            *metric_names,
            writer=self.writer,
            section="train",
            use_wandb=self.use_wandb,
        )
        self.valid_metrics = MetricTracker(
            "loss",
            *metric_names,
            writer=self.writer,
            section="val",
            use_wandb=self.use_wandb,
        )
        self.history = []

        if self.use_wandb:
            wandb_kwargs = {"project": config["name"], "config": config.config}
            if self.resume_ckpt is not None and wandb_id:
                wandb_kwargs["id"] = wandb_id
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)

            wandb.define_metric("global_train_step")
            wandb.define_metric("train_*", step_metric="global_train_step")
            wandb.define_metric("epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("ep/*", step_metric="epoch")

    def _forward(self, data, target):
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            logits = self.model(data)
            loss = self.criterion(logits.float(), target)
        return loss, logits.detach().float()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        n_batches = len(self.data_loader)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss, logits = self._forward(data, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()
            self.writer.set_step(self.global_train_step, "train")
            self.train_metrics.update("loss", loss.item(), step=self.global_train_step)
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(logits, target), step=self.global_train_step
                )
            self.global_train_step += 1

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f"Train Epoch: {epoch} {self._progress(batch_idx, n_batches)} "
                    f"Loss: {loss.item():.6f}"
                )

        log = self.train_metrics.result()
        log["lr"] = self.optimizer.param_groups[0]["lr"]

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None and not isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            self.lr_scheduler.step()

        self._log_epoch(epoch, log)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                loss, logits = self._forward(data, target)

                self.writer.set_step(self.global_val_step, "valid")
                self.global_val_step += 1
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(logits, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.raw_model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _log_epoch(self, epoch, log):
        self.history.append({"epoch": epoch, **{k: float(v) for k, v in log.items()}})
        write_json(self.history, self.checkpoint_dir / "history.json")

        if self.use_wandb:
            payload = {"epoch": epoch}
            for k, v in log.items():
                if k.startswith("val_"):
                    payload[f"val/{k[4:]}"] = float(v)
                else:
                    payload[f"ep/{k}"] = float(v)
            wandb.log(payload)

    @staticmethod
    def _progress(batch_idx, n_batches):
        pct = 100.0 * batch_idx / max(n_batches, 1)
        return f"[{batch_idx}/{n_batches} ({pct:.0f}%)]"
