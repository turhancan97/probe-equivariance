"""Visualize equivariance probe predictions against ground-truth targets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.datasets.builder import build_dataset, build_loader
from evals.datasets.unreal_motion_capture import LINE_MODES, ORBIT_MODES, regression_dim
from evals.models.backbone import load_backbone
from train_equivariance import (
    build_group_loader,
    checkpoint_name,
    collect_split_features,
    compute_rmse,
    is_efficient_probing_probe,
    predict_from_image_loader,
    stack_records,
    validate_probe_backbone_compatibility,
)

SPLIT_PRED_COLORS = {
    "train": "tab:orange",
    "valid": "tab:green",
    "test": "tab:red",
}
GROUND_TRUTH_COLOR = "tab:blue"


def _ensure_split_names(splits: Iterable[str]) -> List[str]:
    normalized = []
    for split in splits:
        split_key = "valid" if split in {"val", "valid"} else split
        if split_key not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        normalized.append(split_key)
    return normalized


def _load_run_config(result_dir: Path, cfg: DictConfig) -> DictConfig:
    run_config_path = Path(cfg.run_config_path) if cfg.run_config_path else result_dir / "run_config.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    run_cfg = OmegaConf.load(run_config_path)
    if cfg.dataset_root is not None:
        run_cfg.dataset.root = cfg.dataset_root
    return run_cfg


def _load_group_head(checkpoint_path: Path, run_cfg: DictConfig, feat_dim: int, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    output_dim = int(payload.get("output_dim", 2))
    head = instantiate(run_cfg.probe, feat_dim=feat_dim, output_dim=output_dim)
    head.load_state_dict(payload["state_dict"])
    head = head.to(device)
    head.eval()
    return head


def _plot_line_ground_truth(ax, targets: np.ndarray) -> None:
    zeros = np.zeros_like(targets)
    ax.scatter(targets, zeros, label="ground truth", c=GROUND_TRUTH_COLOR, alpha=0.5, s=26)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("Normalized line position")
    ax.set_ylabel("Fixed Y=0")


def _plot_line_predictions(ax, preds: np.ndarray, split: str) -> None:
    zeros = np.zeros_like(preds)
    ax.scatter(
        preds,
        zeros,
        label=f"prediction ({split})",
        c=SPLIT_PRED_COLORS[split],
        alpha=0.9,
        marker="x",
        s=32,
    )


def _plot_orbit_ground_truth(ax, targets: np.ndarray) -> None:
    circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", linewidth=1.0, color="0.5")
    ax.add_patch(circle)
    ax.scatter(targets[:, 0], targets[:, 1], label="ground truth", c=GROUND_TRUTH_COLOR, alpha=0.5, s=26)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    ax.set_aspect("equal", adjustable="box")


def _plot_orbit_predictions(ax, preds: np.ndarray, split: str) -> None:
    ax.scatter(
        preds[:, 0],
        preds[:, 1],
        label=f"prediction ({split})",
        c=SPLIT_PRED_COLORS[split],
        alpha=0.9,
        marker="x",
        s=32,
    )


def plot_group_predictions(
    output_path: Path,
    environment: str,
    mode: str,
    object_name: str,
    split_targets: dict[str, torch.Tensor],
    split_preds: dict[str, torch.Tensor],
) -> None:
    gt_tensors = [split_targets[split].detach().cpu() for split in split_targets]
    all_targets = torch.cat(gt_tensors, dim=0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    if mode in LINE_MODES:
        _plot_line_ground_truth(ax, all_targets.numpy()[:, 0])
        for split, preds in split_preds.items():
            _plot_line_predictions(ax, preds.detach().cpu().numpy()[:, 0], split)
    elif mode in ORBIT_MODES:
        _plot_orbit_ground_truth(ax, all_targets.numpy())
        for split, preds in split_preds.items():
            _plot_orbit_predictions(ax, preds.detach().cpu().numpy(), split)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rmse_parts = []
    for split in split_preds:
        rmse = compute_rmse(split_preds[split].detach().cpu(), split_targets[split].detach().cpu())
        rmse_parts.append(f"{split}={rmse:.4f}")

    ax.set_title(
        f"{environment} | {mode} | {object_name}\n" + "RMSE: " + ", ".join(rmse_parts)
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_visualization(cfg: DictConfig) -> None:
    result_dir = Path(cfg.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

    run_cfg = _load_run_config(result_dir, cfg)
    validate_probe_backbone_compatibility(run_cfg)
    uses_ep = is_efficient_probing_probe(run_cfg.probe)
    splits = _ensure_split_names(cfg.splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.output_dir) if cfg.output_dir else result_dir / cfg.output_subdir
    checkpoint_dir = result_dir / cfg.checkpoint_subdir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}. Enable training.save_checkpoints during training first."
        )

    model, feat_dim = load_backbone(
        run_cfg.backbone.name,
        pool=run_cfg.backbone.pool,
        image_mean=run_cfg.backbone.get("image_mean"),
        custom_mean=run_cfg.backbone.get("custom_mean"),
        custom_std=run_cfg.backbone.get("custom_std"),
    )
    model = model.to(device)
    norm_overrides = {}
    if getattr(model, "normalize_mean", None) is not None and getattr(model, "normalize_std", None) is not None:
        norm_overrides = {"mean": model.normalize_mean, "std": model.normalize_std}

    if uses_ep:
        split_datasets: dict[str, object] = {}
        all_group_keys = set()
        for split in splits:
            if split == "test" and float(run_cfg.dataset.test_ratio) <= 0:
                logger.info("Skipping test split because dataset.test_ratio <= 0")
                continue

            dataset = build_dataset(run_cfg.dataset, split, **norm_overrides)
            split_datasets[split] = dataset
            group_keys = dataset.group_indices_by_split[split].keys()
            all_group_keys.update(group_keys)
            if not group_keys:
                logger.info(f"No samples found for split: {split}")

        for environment, mode, object_name in sorted(all_group_keys):
            checkpoint_path = checkpoint_dir / checkpoint_name(environment, mode, object_name)
            if not checkpoint_path.exists():
                logger.warning(f"Missing checkpoint for {environment}/{mode}/{object_name}: {checkpoint_path}")
                continue

            head = _load_group_head(checkpoint_path, run_cfg, feat_dim, device)
            split_targets: dict[str, torch.Tensor] = {}
            split_preds: dict[str, torch.Tensor] = {}

            for split in splits:
                dataset = split_datasets.get(split)
                if dataset is None:
                    continue

                loader, _ = build_group_loader(dataset, split, (environment, mode, object_name), run_cfg.batch_size)
                if loader is None:
                    continue

                preds, targets = predict_from_image_loader(head, model, loader, device)
                if preds.numel() == 0:
                    continue

                expected_dim = regression_dim(mode)
                if preds.shape[1] != expected_dim:
                    raise ValueError(
                        f"Checkpoint output dim mismatch for {environment}/{mode}/{object_name}: "
                        f"expected {expected_dim}, got {preds.shape[1]}"
                    )

                split_targets[split] = targets
                split_preds[split] = preds

            if not split_preds:
                continue

            filename = checkpoint_name(environment, mode, object_name).replace('.pt', '.png')
            plot_group_predictions(
                output_dir / filename,
                environment,
                mode,
                object_name,
                split_targets,
                split_preds,
            )
        return

    groups_by_split: dict[str, dict] = {}
    all_group_keys = set()
    for split in splits:
        if split == "test" and float(run_cfg.dataset.test_ratio) <= 0:
            logger.info("Skipping test split because dataset.test_ratio <= 0")
            continue

        loader = build_loader(run_cfg.dataset, split, run_cfg.batch_size, **norm_overrides)
        groups = collect_split_features(loader, model, device)
        groups_by_split[split] = groups
        all_group_keys.update(groups.keys())
        if not groups:
            logger.info(f"No samples found for split: {split}")

    for environment, mode, object_name in sorted(all_group_keys):
        checkpoint_path = checkpoint_dir / checkpoint_name(environment, mode, object_name)
        if not checkpoint_path.exists():
            logger.warning(f"Missing checkpoint for {environment}/{mode}/{object_name}: {checkpoint_path}")
            continue

        head = _load_group_head(checkpoint_path, run_cfg, feat_dim, device)
        split_targets: dict[str, torch.Tensor] = {}
        split_preds: dict[str, torch.Tensor] = {}

        for split in splits:
            groups = groups_by_split.get(split, {})
            records = groups.get((environment, mode, object_name), [])
            if not records:
                continue

            features, targets = stack_records(records)
            if features.numel() == 0:
                continue

            with torch.no_grad():
                preds = head(features.to(device)).detach().cpu()

            expected_dim = regression_dim(mode)
            if preds.shape[1] != expected_dim:
                raise ValueError(
                    f"Checkpoint output dim mismatch for {environment}/{mode}/{object_name}: "
                    f"expected {expected_dim}, got {preds.shape[1]}"
                )

            split_targets[split] = targets
            split_preds[split] = preds

        if not split_preds:
            continue

        filename = checkpoint_name(environment, mode, object_name).replace('.pt', '.png')
        plot_group_predictions(
            output_dir / filename,
            environment,
            mode,
            object_name,
            split_targets,
            split_preds,
        )


@hydra.main(config_name="equivariance_visualization", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    run_visualization(cfg)


if __name__ == "__main__":
    main()
