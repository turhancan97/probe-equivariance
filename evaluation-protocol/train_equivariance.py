"""Training script for equivariance regression probes with a frozen backbone,
fit to unreal-motion-capture's actual recording layout. Single-GPU only.
"""

from __future__ import annotations

import json
import math
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from evals.datasets.builder import build_dataset, build_group_loader_from_dataset, build_loader
from evals.datasets.unreal_motion_capture import regression_dim
from evals.models.backbone import load_backbone
from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed


EP_PROBE_TARGET = "evals.models.probe.EfficientProbingHead"

OBJECT_METRIC_FIELDS = [
    "Timestamp",
    "Seed",
    "Experiment",
    "Environment",
    "Mode",
    "Object",
    "Train RMSE",
    "Val RMSE",
    "Test RMSE",
    "Num Train",
    "Num Val",
    "Num Test",
    "Backbone",
    "Pool",
    "Head",
]


def _ensure_seed_column(object_csv: Path, seed: int) -> None:
    """Migrate legacy object metrics before appending seed-aware rows."""
    if not object_csv.exists():
        return

    with object_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "Seed" in (reader.fieldnames or []):
            return
        legacy_rows = list(reader)

    with object_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBJECT_METRIC_FIELDS)
        writer.writeheader()
        for row in legacy_rows:
            row["Seed"] = str(seed)
            writer.writerow({field: row.get(field, "") for field in OBJECT_METRIC_FIELDS})


def _instantiate_probe(cfg: DictConfig, feat_dim: int, output_dim: int):
    """Instantiate cfg.probe, unwrapping Hydra's InstantiationException so config
    validation errors (e.g. ValueError from EfficientProbingPool) propagate as-is."""
    try:
        return instantiate(cfg.probe, feat_dim=feat_dim, output_dim=output_dim)
    except hydra.errors.InstantiationException as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise


def _sanitize_name(name: str) -> str:
    sanitized = name.replace("/", "_").replace(" ", "_").replace("-", "_")
    return "".join(c for c in sanitized if c.isalnum() or c == "_").lower()


def checkpoint_name(environment: str, mode: str, object_name: str) -> str:
    return "__".join(
        [_sanitize_name(environment), _sanitize_name(mode), _sanitize_name(object_name)]
    ) + ".pt"


@dataclass
class FeatureRecord:
    feature: torch.Tensor
    target: torch.Tensor
    frame_index: int


def is_efficient_probing_probe(probe_cfg: DictConfig) -> bool:
    return str(probe_cfg.get("_target_", "")) == EP_PROBE_TARGET


def validate_probe_backbone_compatibility(cfg: DictConfig) -> None:
    uses_ep = is_efficient_probing_probe(cfg.probe)
    pool = str(cfg.backbone.pool)
    if uses_ep and pool != "patch":
        raise ValueError("Efficient Probing requires backbone.pool='patch'.")
    if not uses_ep and pool == "patch":
        raise ValueError("backbone.pool='patch' is currently supported only with Efficient Probing.")


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


def _sorted_group_sample_indices(dataset, split: str, key: Tuple[str, str, str]) -> List[int]:
    return sorted(
        dataset.group_indices_by_split[split].get(key, []),
        key=lambda sample_idx: dataset.samples[sample_idx].frame_index,
    )


def build_group_loader(
    dataset,
    split: str,
    key: Tuple[str, str, str],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
):
    sample_indices = _sorted_group_sample_indices(dataset, split, key)
    if not sample_indices:
        return None, []
    loader = build_group_loader_from_dataset(
        dataset,
        sample_indices,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader, sample_indices


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


def predict_from_image_loader(
    head,
    model,
    loader: DataLoader | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if loader is None:
        return torch.empty(0), torch.empty(0)

    preds_batches = []
    target_batches = []
    head.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].detach().cpu().contiguous()
            feats = model(images)
            preds = head(feats).detach().cpu().contiguous()
            preds_batches.append(preds)
            target_batches.append(targets)

    if not preds_batches:
        return torch.empty(0), torch.empty(0)

    return torch.cat(preds_batches, dim=0), torch.cat(target_batches, dim=0)


def evaluate_head_on_image_loader(head, model, loader: DataLoader | None, device: torch.device) -> Tuple[float, float]:
    preds, targets = predict_from_image_loader(head, model, loader, device)
    if preds.numel() == 0:
        return float("nan"), float("nan")
    mse = torch.mean((preds - targets) ** 2).item()
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
        return {
            "train_mse": float("nan"),
            "train_rmse": float("nan"),
            "val_mse": float("nan"),
            "val_rmse": float("nan"),
            "head": head,
        }

    head = head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.optimizer.probe_lr, weight_decay=cfg.optimizer.weight_decay
    )

    steps_per_epoch = max(1, math.ceil(train_features.size(0) / max(1, cfg.batch_size)))
    total_steps = cfg.optimizer.n_epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.optimizer.warmup_epochs * steps_per_epoch))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, warmup_steps),
    )
    loss_fn = torch.nn.MSELoss()

    indices = torch.arange(train_features.size(0))
    val_mse_epoch = float("nan")
    epoch_bar = tqdm(range(cfg.optimizer.n_epochs), desc=f"Epochs[{log_prefix}]", leave=False)
    for epoch in epoch_bar:
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

        epoch_bar.set_postfix(train_mse=train_mse_epoch, val_mse=val_mse_epoch)
        if log_to_wandb:
            wandb.log({f"train_mse_{log_prefix}": train_mse_epoch, f"val_mse_{log_prefix}": val_mse_epoch})

    head.eval()
    with torch.no_grad():
        train_preds = head(train_features.to(device)).detach().cpu()
    train_rmse = compute_rmse(train_preds, train_targets)
    train_mse = float(torch.mean((train_preds - train_targets) ** 2).item())
    val_mse, val_rmse = evaluate_head(head, val_features, val_targets, device)

    return {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "head": head,
    }


def train_probe_on_image_loaders(
    head,
    model,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    device: torch.device,
    cfg: DictConfig,
    log_prefix: str,
    log_to_wandb: bool,
) -> Dict[str, float]:
    train_dataset_size = len(train_loader.dataset) if train_loader is not None else 0
    if train_dataset_size == 0:
        return {
            "train_mse": float("nan"),
            "train_rmse": float("nan"),
            "val_mse": float("nan"),
            "val_rmse": float("nan"),
            "head": head,
        }

    head = head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.optimizer.probe_lr, weight_decay=cfg.optimizer.weight_decay
    )

    steps_per_epoch = max(1, math.ceil(train_dataset_size / max(1, cfg.batch_size)))
    total_steps = cfg.optimizer.n_epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.optimizer.warmup_epochs * steps_per_epoch))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, warmup_steps),
    )
    loss_fn = torch.nn.MSELoss()

    val_mse_epoch = float("nan")
    epoch_bar = tqdm(range(cfg.optimizer.n_epochs), desc=f"Epochs[{log_prefix}]", leave=False)
    for epoch in epoch_bar:
        head.train()
        epoch_loss, count = 0.0, 0
        for batch in tqdm(train_loader, desc="Batches", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.no_grad():
                feats = model(images)
            preds = head(feats)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * targets.size(0)
            count += targets.size(0)

        train_mse_epoch = epoch_loss / max(1, count)
        if cfg.training.eval_every_epochs > 0 and epoch % cfg.training.eval_every_epochs == 0:
            val_mse_epoch, _ = evaluate_head_on_image_loader(head, model, val_loader, device)

        epoch_bar.set_postfix(train_mse=train_mse_epoch, val_mse=val_mse_epoch)
        if log_to_wandb:
            wandb.log({f"train_mse_{log_prefix}": train_mse_epoch, f"val_mse_{log_prefix}": val_mse_epoch})

    train_preds, train_targets = predict_from_image_loader(head, model, train_loader, device)
    train_rmse = compute_rmse(train_preds, train_targets)
    train_mse = float(torch.mean((train_preds - train_targets) ** 2).item()) if train_preds.numel() else float("nan")
    val_mse, val_rmse = evaluate_head_on_image_loader(head, model, val_loader, device)

    return {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "head": head,
    }


def save_probe_checkpoint(
    checkpoint_path: Path,
    head,
    feat_dim: int,
    output_dim: int,
    environment: str,
    mode: str,
    object_name: str,
    cfg: DictConfig,
) -> None:
    payload = {
        "environment": environment,
        "mode": mode,
        "object": object_name,
        "feat_dim": feat_dim,
        "output_dim": output_dim,
        "backbone": {
            "name": cfg.backbone.name,
            "pool": cfg.backbone.pool,
        },
        "probe": OmegaConf.to_container(cfg.probe, resolve=True),
        "state_dict": head.cpu().state_dict(),
    }
    torch.save(payload, checkpoint_path)


def run_equivariance(cfg: DictConfig) -> None:
    validate_probe_backbone_compatibility(cfg)
    set_random_seed(cfg.system.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = bool(cfg.wandb.use)
    uses_ep = is_efficient_probing_probe(cfg.probe)

    exp_path = Path(__file__).parent / f"equivariance_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
    exp_path.mkdir(parents=True, exist_ok=True)
    logger.add(exp_path / "training.log")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    result_dir = Path(cfg.output_dir) / f"equivariance_{cfg.experiment_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoints = bool(cfg.training.save_checkpoints)
    checkpoint_dir = result_dir / "checkpoints"
    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb and wandb is None:
        raise ModuleNotFoundError("wandb.use=true, but wandb is not installed.")

    requested_image_size = cfg.dataset.get("image_size")
    if requested_image_size is not None and int(requested_image_size) <= 0:
        raise ValueError("dataset.image_size must be positive when specified")

    model, feat_dim = load_backbone(
        cfg.backbone.name,
        pool=cfg.backbone.pool,
        image_mean=cfg.backbone.get("image_mean"),
        custom_mean=cfg.backbone.get("custom_mean"),
        custom_std=cfg.backbone.get("custom_std"),
        img_size=None if requested_image_size is None else int(requested_image_size),
    )
    model = model.to(device)
    resolved_image_size = (
        int(requested_image_size)
        if requested_image_size is not None
        else int(getattr(model, "input_size", 224))
    )
    cfg.dataset.image_size = resolved_image_size
    (result_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    if use_wandb:
        wandb.init(
            project="probe-equivariance",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group=f"seed:{cfg.system.random_seed}",
        )
    norm_overrides = {}
    if getattr(model, "normalize_mean", None) is not None and getattr(model, "normalize_std", None) is not None:
        norm_overrides = {"mean": model.normalize_mean, "std": model.normalize_std}

    if uses_ep:
        num_workers = cfg.training.get("num_workers", 4)
        train_loader = build_loader(
            cfg.dataset, "train", cfg.batch_size, num_workers=num_workers, **norm_overrides
        )
        val_loader = build_loader(
            cfg.dataset, "valid", cfg.batch_size, num_workers=num_workers, **norm_overrides
        )
        test_loader = (
            build_loader(cfg.dataset, "test", cfg.batch_size, num_workers=num_workers, **norm_overrides)
            if cfg.dataset.test_ratio > 0
            else None
        )

        logger.info("Caching Efficient Probing patch-token features in CPU memory for this run")
        train_groups = collect_split_features(train_loader, model, device)
        val_groups = collect_split_features(val_loader, model, device)
        test_groups = collect_split_features(test_loader, model, device) if test_loader else {}

        keys = sorted(train_groups.keys())
        backbone_name = model.checkpoint_name
        head_name = cfg.probe.get("_target_", "probe")
        pool_name = cfg.backbone.get("pool", "")
        results = []
        checkpoint_manifest = []

        for key in tqdm(keys, desc="Groups"):
            environment, mode, object_name = key
            train_records = train_groups.get(key, [])
            val_records = val_groups.get(key, [])
            test_records = test_groups.get(key, [])
            log_prefix = "_".join([
                _sanitize_name(environment),
                _sanitize_name(mode),
                _sanitize_name(object_name),
            ])

            if len(train_records) < cfg.training.min_samples_per_object:
                logger.info(
                    f"Skipping {environment}/{mode}/{object_name} (insufficient samples: {len(train_records)})"
                )
                results.append(
                    {
                        "environment": environment,
                        "mode": mode,
                        "object": object_name,
                        "train_mse": float("nan"),
                        "val_mse": float("nan"),
                        "test_mse": float("nan"),
                        "train_rmse": float("nan"),
                        "val_rmse": float("nan"),
                        "test_rmse": float("nan"),
                        "num_train": len(train_records),
                        "num_val": len(val_records),
                        "num_test": len(test_records),
                    }
                )
                checkpoint_manifest.append(
                    {
                        "environment": environment,
                        "mode": mode,
                        "object": object_name,
                        "checkpoint_path": None,
                        "num_train": len(train_records),
                        "num_val": len(val_records),
                        "num_test": len(test_records),
                    }
                )
                continue

            output_dim = regression_dim(mode)
            train_features, train_targets = stack_records(train_records)
            val_features, val_targets = stack_records(val_records)
            test_features, test_targets = stack_records(test_records)
            head = _instantiate_probe(cfg, feat_dim, output_dim)
            metrics = train_probe(
                head,
                train_features,
                train_targets,
                val_features,
                val_targets,
                device,
                cfg,
                log_prefix,
                use_wandb,
            )
            trained_head = metrics.pop("head")
            test_mse, test_rmse = evaluate_head(trained_head, test_features, test_targets, device)

            checkpoint_relpath = None
            if save_checkpoints:
                checkpoint_path = checkpoint_dir / checkpoint_name(environment, mode, object_name)
                save_probe_checkpoint(
                    checkpoint_path,
                    trained_head,
                    feat_dim,
                    output_dim,
                    environment,
                    mode,
                    object_name,
                    cfg,
                )
                checkpoint_relpath = checkpoint_path.relative_to(result_dir).as_posix()

            results.append(
                {
                    "environment": environment,
                    "mode": mode,
                    "object": object_name,
                    "train_mse": metrics["train_mse"],
                    "val_mse": metrics["val_mse"],
                    "test_mse": test_mse,
                    "train_rmse": metrics["train_rmse"],
                    "val_rmse": metrics["val_rmse"],
                    "test_rmse": test_rmse,
                    "num_train": len(train_records),
                    "num_val": len(val_records),
                    "num_test": len(test_records),
                }
            )
            checkpoint_manifest.append(
                {
                    "environment": environment,
                    "mode": mode,
                    "object": object_name,
                    "checkpoint_path": checkpoint_relpath,
                    "num_train": len(train_records),
                    "num_val": len(val_records),
                    "num_test": len(test_records),
                }
            )
    else:
        num_workers = cfg.training.get("num_workers", 4)
        train_loader = build_loader(cfg.dataset, "train", cfg.batch_size, num_workers=num_workers, **norm_overrides)
        val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size, num_workers=num_workers, **norm_overrides)
        test_loader = (
            build_loader(cfg.dataset, "test", cfg.batch_size, num_workers=num_workers, **norm_overrides)
            if cfg.dataset.test_ratio > 0
            else None
        )

        train_groups = collect_split_features(train_loader, model, device)
        val_groups = collect_split_features(val_loader, model, device)
        test_groups = collect_split_features(test_loader, model, device) if test_loader else {}

        backbone_name = model.checkpoint_name
        head_name = cfg.probe.get("_target_", "probe")
        pool_name = cfg.backbone.get("pool", "")

        results = []
        checkpoint_manifest = []
        keys = sorted(train_groups.keys())

        for key in tqdm(keys, desc="Groups"):
            environment, mode, object_name = key
            train_records = train_groups.get(key, [])
            val_records = val_groups.get(key, [])
            test_records = test_groups.get(key, [])
            log_prefix = "_".join([
                _sanitize_name(environment),
                _sanitize_name(mode),
                _sanitize_name(object_name),
            ])

            if len(train_records) < cfg.training.min_samples_per_object:
                logger.info(
                    f"Skipping {environment}/{mode}/{object_name} (insufficient samples: {len(train_records)})"
                )
                results.append(
                    {
                        "environment": environment,
                        "mode": mode,
                        "object": object_name,
                        "train_mse": float("nan"),
                        "val_mse": float("nan"),
                        "test_mse": float("nan"),
                        "train_rmse": float("nan"),
                        "val_rmse": float("nan"),
                        "test_rmse": float("nan"),
                        "num_train": len(train_records),
                        "num_val": len(val_records),
                        "num_test": len(test_records),
                    }
                )
                checkpoint_manifest.append(
                    {
                        "environment": environment,
                        "mode": mode,
                        "object": object_name,
                        "checkpoint_path": None,
                        "num_train": len(train_records),
                        "num_val": len(val_records),
                        "num_test": len(test_records),
                    }
                )
                continue

            train_features, train_targets = stack_records(train_records)
            val_features, val_targets = stack_records(val_records)
            test_features, test_targets = stack_records(test_records)

            output_dim = regression_dim(mode)
            head = _instantiate_probe(cfg, feat_dim, output_dim)

            metrics = train_probe(
                head, train_features, train_targets, val_features, val_targets, device, cfg, log_prefix, use_wandb
            )
            trained_head = metrics.pop("head")
            test_mse, test_rmse = evaluate_head(trained_head, test_features, test_targets, device)

            checkpoint_relpath = None
            if save_checkpoints:
                checkpoint_path = checkpoint_dir / checkpoint_name(environment, mode, object_name)
                save_probe_checkpoint(
                    checkpoint_path,
                    trained_head,
                    feat_dim,
                    output_dim,
                    environment,
                    mode,
                    object_name,
                    cfg,
                )
                checkpoint_relpath = checkpoint_path.relative_to(result_dir).as_posix()

            results.append(
                {
                    "environment": environment,
                    "mode": mode,
                    "object": object_name,
                    "train_mse": metrics["train_mse"],
                    "val_mse": metrics["val_mse"],
                    "test_mse": test_mse,
                    "train_rmse": metrics["train_rmse"],
                    "val_rmse": metrics["val_rmse"],
                    "test_rmse": test_rmse,
                    "num_train": len(train_records),
                    "num_val": len(val_records),
                    "num_test": len(test_records),
                }
            )
            checkpoint_manifest.append(
                {
                    "environment": environment,
                    "mode": mode,
                    "object": object_name,
                    "checkpoint_path": checkpoint_relpath,
                    "num_train": len(train_records),
                    "num_val": len(val_records),
                    "num_test": len(test_records),
                }
            )

    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    seed = int(cfg.system.random_seed)

    object_csv = result_dir / "object_metrics.csv"
    _ensure_seed_column(object_csv, seed)
    with object_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OBJECT_METRIC_FIELDS)
        if object_csv.stat().st_size == 0:
            writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "Timestamp": timestamp,
                    "Seed": seed,
                    "Experiment": cfg.experiment_name,
                    "Environment": row["environment"],
                    "Mode": row["mode"],
                    "Object": row["object"],
                    "Train RMSE": row["train_rmse"],
                    "Val RMSE": row["val_rmse"],
                    "Test RMSE": row["test_rmse"],
                    "Num Train": row["num_train"],
                    "Num Val": row["num_val"],
                    "Num Test": row["num_test"],
                    "Backbone": backbone_name,
                    "Pool": pool_name,
                    "Head": head_name,
                }
            )

    environments = sorted({r["environment"] for r in results})
    modes = sorted({r["mode"] for r in results})
    summary_csv = result_dir / "mode_summary.csv"
    new_summary = not summary_csv.exists()
    with summary_csv.open("a") as f:
        if new_summary:
            header = ["Timestamp", "Experiment", "Environment", "Backbone", "Pool", "Head"]
            header += [f"Train RMSE {m}" for m in modes] + [f"Val RMSE {m}" for m in modes] + [f"Test RMSE {m}" for m in modes]
            f.write(",".join(header) + "\n")

        for environment in environments:
            row_values = [timestamp, cfg.experiment_name, environment, backbone_name, pool_name, head_name]
            for metric_key in ("train_rmse", "val_rmse", "test_rmse"):
                for mode in modes:
                    vals = [r[metric_key] for r in results if r["environment"] == environment and r["mode"] == mode]
                    avg = float(torch.tensor(vals).nanmean().item()) if vals else float("nan")
                    row_values.append(str(avg))
            f.write(",".join(row_values) + "\n")

    manifest_payload = {
        "timestamp": timestamp,
        "experiment_name": cfg.experiment_name,
        "backbone": backbone_name,
        "pool": pool_name,
        "head": head_name,
        "save_checkpoints": save_checkpoints,
        "groups": checkpoint_manifest,
    }
    (result_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    if use_wandb:
        wandb.finish()


@hydra.main(config_name="equivariance_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    run_equivariance(cfg)


if __name__ == "__main__":
    main()
