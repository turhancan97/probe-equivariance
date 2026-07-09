"""Training script for equivariance regression probes with a frozen backbone,
fit to unreal-motion-capture's actual recording layout. Single-GPU only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import hydra
import torch
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.datasets.unreal_motion_capture import regression_dim
from evals.models.backbone import load_backbone
from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed


def _sanitize_name(name: str) -> str:
    sanitized = name.replace("/", "_").replace(" ", "_").replace("-", "_")
    return "".join(c for c in sanitized if c.isalnum() or c == "_").lower()


@dataclass
class FeatureRecord:
    feature: torch.Tensor
    target: torch.Tensor
    frame_index: int


def collect_split_features(
    loader, model, device: torch.device
) -> Dict[Tuple[str, str, str], List[FeatureRecord]]:
    groups: DefaultDict[Tuple[str, str, str], List[FeatureRecord]] = defaultdict(list)
    if loader is None:
        return groups

    for batch in tqdm(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"]
        environments = batch["environment"]
        modes = batch["mode"]
        objects = batch["object_class"]
        frame_indices = batch["frame_index"].tolist()

        with torch.no_grad():
            feats = model(images)

        feats_cpu = feats.detach().cpu().contiguous()
        targets_cpu = targets.detach().cpu().contiguous()

        for idx in range(feats_cpu.size(0)):
            key = (environments[idx], modes[idx], objects[idx])
            groups[key].append(
                FeatureRecord(
                    feature=feats_cpu[idx].clone(),
                    target=targets_cpu[idx].clone(),
                    frame_index=int(frame_indices[idx]),
                )
            )

    return groups


def stack_records(records: List[FeatureRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not records:
        return torch.empty(0), torch.empty(0)
    records_sorted = sorted(records, key=lambda r: r.frame_index)
    features = torch.stack([r.feature for r in records_sorted])
    targets = torch.stack([r.target for r in records_sorted])
    return features, targets


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if preds.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean((preds - targets) ** 2)).item())


def evaluate_head(head, features: torch.Tensor, targets: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    if features.numel() == 0:
        return float("nan"), float("nan")
    head.eval()
    with torch.no_grad():
        preds = head(features.to(device))
        mse = torch.mean((preds - targets.to(device)) ** 2).item()
    return mse, math.sqrt(mse) if mse >= 0 else float("nan")


def train_probe(
    head,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    cfg: DictConfig,
    log_prefix: str,
    log_to_wandb: bool,
) -> Dict[str, float]:
    if train_features.numel() == 0:
        return {"train_mse": float("nan"), "train_rmse": float("nan"), "val_mse": float("nan"), "val_rmse": float("nan"), "head": head}

    head = head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.optimizer.probe_lr, weight_decay=cfg.optimizer.weight_decay
    )

    steps_per_epoch = max(1, math.ceil(train_features.size(0) / max(1, cfg.batch_size)))
    total_steps = cfg.optimizer.n_epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.optimizer.warmup_epochs * steps_per_epoch))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, warmup_steps))
    loss_fn = torch.nn.MSELoss()

    indices = torch.arange(train_features.size(0))
    val_mse_epoch = float("nan")
    for epoch in range(cfg.optimizer.n_epochs):
        head.train()
        perm = indices[torch.randperm(indices.size(0))]
        epoch_loss, count = 0.0, 0
        for batch_indices in perm.split(max(1, cfg.batch_size)):
            feats = train_features[batch_indices].to(device, non_blocking=True)
            targets = train_targets[batch_indices].to(device, non_blocking=True)
            preds = head(feats)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * batch_indices.size(0)
            count += batch_indices.size(0)

        train_mse_epoch = epoch_loss / max(1, count)
        if cfg.training.eval_every_epochs > 0 and epoch % cfg.training.eval_every_epochs == 0:
            val_mse_epoch, _ = evaluate_head(head, val_features, val_targets, device)

        if log_to_wandb:
            wandb.log({f"train_mse_{log_prefix}": train_mse_epoch, f"val_mse_{log_prefix}": val_mse_epoch})

    head.eval()
    with torch.no_grad():
        train_preds = head(train_features.to(device)).detach().cpu()
    train_rmse = compute_rmse(train_preds, train_targets)
    train_mse = float(torch.mean((train_preds - train_targets) ** 2).item())
    val_mse, val_rmse = evaluate_head(head, val_features, val_targets, device)

    return {"train_mse": train_mse, "train_rmse": train_rmse, "val_mse": val_mse, "val_rmse": val_rmse, "head": head}


def run_equivariance(cfg: DictConfig) -> None:
    set_random_seed(cfg.system.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_path = Path(__file__).parent / f"equivariance_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
    exp_path.mkdir(parents=True, exist_ok=True)
    logger.add(exp_path / "training.log")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    if cfg.wandb.use:
        wandb.init(
            project="probe-equivariance",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group=f"seed:{cfg.system.random_seed}",
        )

    train_loader = build_loader(cfg.dataset, "train", cfg.batch_size)
    val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size) if cfg.dataset.test_ratio > 0 else None

    model, feat_dim = load_backbone(cfg.backbone.name, pool=cfg.backbone.pool)
    model = model.to(device)

    train_groups = collect_split_features(train_loader, model, device)
    val_groups = collect_split_features(val_loader, model, device)
    test_groups = collect_split_features(test_loader, model, device) if test_loader else {}

    backbone_name = model.checkpoint_name
    head_name = cfg.probe.get("_target_", "probe")

    results = []
    keys = sorted(train_groups.keys())

    for key in tqdm(keys, desc="Groups"):
        environment, mode, object_name = key
        train_records = train_groups.get(key, [])
        val_records = val_groups.get(key, [])
        test_records = test_groups.get(key, [])

        if len(train_records) < cfg.training.min_samples_per_object:
            logger.info(f"Skipping {environment}/{mode}/{object_name} (insufficient samples: {len(train_records)})")
            results.append(
                {
                    "environment": environment, "mode": mode, "object": object_name,
                    "train_mse": float("nan"), "val_mse": float("nan"), "test_mse": float("nan"),
                    "train_rmse": float("nan"), "val_rmse": float("nan"), "test_rmse": float("nan"),
                    "num_train": len(train_records), "num_val": len(val_records), "num_test": len(test_records),
                }
            )
            continue

        train_features, train_targets = stack_records(train_records)
        val_features, val_targets = stack_records(val_records)
        test_features, test_targets = stack_records(test_records)

        output_dim = regression_dim(mode)
        head = instantiate(cfg.probe, feat_dim=feat_dim, output_dim=output_dim)

        log_prefix = "_".join([_sanitize_name(environment), _sanitize_name(mode), _sanitize_name(object_name)])

        metrics = train_probe(
            head, train_features, train_targets, val_features, val_targets, device, cfg, log_prefix, cfg.wandb.use
        )
        trained_head = metrics.pop("head")
        test_mse, test_rmse = evaluate_head(trained_head, test_features, test_targets, device)

        results.append(
            {
                "environment": environment, "mode": mode, "object": object_name,
                "train_mse": metrics["train_mse"], "val_mse": metrics["val_mse"], "test_mse": test_mse,
                "train_rmse": metrics["train_rmse"], "val_rmse": metrics["val_rmse"], "test_rmse": test_rmse,
                "num_train": len(train_records), "num_val": len(val_records), "num_test": len(test_records),
            }
        )

    result_dir = Path(cfg.output_dir) / f"equivariance_{cfg.experiment_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")

    object_csv = result_dir / "object_metrics.csv"
    new_file = not object_csv.exists()
    with object_csv.open("a") as f:
        if new_file:
            f.write("Timestamp,Experiment,Environment,Mode,Object,Train RMSE,Val RMSE,Test RMSE,Num Train,Num Val,Num Test,Backbone,Head\n")
        for row in results:
            f.write(
                f"{timestamp},{cfg.experiment_name},{row['environment']},{row['mode']},{row['object']},"
                f"{row['train_rmse']},{row['val_rmse']},{row['test_rmse']},"
                f"{row['num_train']},{row['num_val']},{row['num_test']},{backbone_name},{head_name}\n"
            )

    environments = sorted({r["environment"] for r in results})
    modes = sorted({r["mode"] for r in results})
    summary_csv = result_dir / "mode_summary.csv"
    new_summary = not summary_csv.exists()
    with summary_csv.open("a") as f:
        if new_summary:
            header = ["Timestamp", "Experiment", "Environment", "Backbone", "Head"]
            header += [f"Train RMSE {m}" for m in modes] + [f"Val RMSE {m}" for m in modes] + [f"Test RMSE {m}" for m in modes]
            f.write(",".join(header) + "\n")

        for environment in environments:
            row_values = [timestamp, cfg.experiment_name, environment, backbone_name, head_name]
            for metric_key in ("train_rmse", "val_rmse", "test_rmse"):
                for mode in modes:
                    vals = [r[metric_key] for r in results if r["environment"] == environment and r["mode"] == mode]
                    avg = float(torch.tensor(vals).nanmean().item()) if vals else float("nan")
                    row_values.append(str(avg))
            f.write(",".join(row_values) + "\n")

    if cfg.wandb.use:
        wandb.finish()


@hydra.main(config_name="equivariance_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    run_equivariance(cfg)


if __name__ == "__main__":
    main()
