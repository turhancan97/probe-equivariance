"""Visualize per-frame backbone representations from one motion directory."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import hydra
import matplotlib
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.models.backbone import get_model_input_size, load_backbone


REPRESENTATION_TO_POOL = {
    "cls": "cls",
    "patch_mean": "mean",
}


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    image_path: Path


class MotionFrameDataset(Dataset):
    """Load all frames in one generated motion directory in pose-JSON order."""

    def __init__(
        self,
        video_dir: str | Path,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.video_dir = Path(video_dir).expanduser()
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory does not exist: {self.video_dir}")
        if not self.video_dir.is_dir():
            raise NotADirectoryError(f"Video path is not a directory: {self.video_dir}")

        self.records = self._load_records()
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def _load_records(self) -> list[FrameRecord]:
        mode_name = self.video_dir.name
        pose_candidates = [
            self.video_dir / f"{mode_name}_camera_poses.json",
            self.video_dir / f"{mode_name}_object_poses.json",
        ]
        pose_paths = [path for path in pose_candidates if path.exists()]
        if len(pose_paths) != 1:
            expected = ", ".join(str(path) for path in pose_candidates)
            raise FileNotFoundError(
                f"Expected exactly one motion pose JSON in {self.video_dir}; checked: {expected}"
            )

        import json

        with pose_paths[0].open("r", encoding="utf-8") as handle:
            poses = json.load(handle)
        if not isinstance(poses, list) or not poses:
            raise ValueError(f"Pose JSON must contain a non-empty list: {pose_paths[0]}")

        records = []
        for frame_index, entry in enumerate(poses):
            if not isinstance(entry, dict) or not isinstance(entry.get("frame"), str):
                raise ValueError(
                    f"Pose entry {frame_index} in {pose_paths[0]} has no string 'frame' field"
                )
            image_path = (self.video_dir / entry["frame"]).resolve()
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Frame {frame_index} references missing image: {image_path}"
                )
            records.append(FrameRecord(frame_index=frame_index, image_path=image_path))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path).convert("RGB") as image:
            image_tensor = self.transform(image.copy())
        return {
            "image": image_tensor,
            "frame_index": record.frame_index,
            "image_path": str(record.image_path),
        }


def representation_pool(representation: str) -> str:
    try:
        return REPRESENTATION_TO_POOL[representation.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(REPRESENTATION_TO_POOL))
        raise ValueError(f"Unsupported representation: {representation!r}; expected one of: {choices}") from error


def _model_input_size(model: Any) -> int:
    return get_model_input_size(getattr(model, "model", model))


def _adaptive_tsne_perplexity(num_samples: int) -> float:
    return float(max(1, min(30, (num_samples - 1) // 3)))


def _adaptive_umap_neighbors(num_samples: int) -> int:
    return max(2, min(15, num_samples - 1))


def reduce_features(features: np.ndarray, cfg: DictConfig) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("At least two video frames are required for 2D representation visualization")
    if features.shape[1] < 2:
        raise ValueError("Backbone representations must have at least two feature dimensions")

    method = str(cfg.reduction.method).lower()
    seed = int(cfg.random_seed)
    if method == "pca":
        pca_cfg = dict(OmegaConf.to_container(cfg.reduction.pca, resolve=True) or {})
        pca_cfg.pop("n_components", None)
        return PCA(n_components=2, random_state=seed, **pca_cfg).fit_transform(features)

    if method == "tsne":
        tsne_cfg = dict(OmegaConf.to_container(cfg.reduction.tsne, resolve=True) or {})
        perplexity = tsne_cfg.pop("perplexity", None)
        perplexity = (
            _adaptive_tsne_perplexity(features.shape[0]) if perplexity is None else float(perplexity)
        )
        if perplexity >= features.shape[0]:
            raise ValueError(
                f"t-SNE perplexity must be smaller than the number of frames ({features.shape[0]}), "
                f"got {perplexity}"
            )
        tsne_cfg.pop("n_components", None)
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            **tsne_cfg,
        ).fit_transform(features)

    if method == "umap":
        if features.shape[0] < 3:
            raise ValueError("UMAP requires at least three video frames")
        try:
            import umap
        except ImportError as error:
            raise ModuleNotFoundError(
                "UMAP support requires the optional 'umap-learn' package. "
                "Install it in the active environment with: pip install umap-learn"
            ) from error

        umap_cfg = dict(OmegaConf.to_container(cfg.reduction.umap, resolve=True) or {})
        n_neighbors = umap_cfg.pop("n_neighbors", None)
        n_neighbors = (
            _adaptive_umap_neighbors(features.shape[0]) if n_neighbors is None else int(n_neighbors)
        )
        if n_neighbors >= features.shape[0]:
            n_neighbors = features.shape[0] - 1
        umap_cfg.pop("n_components", None)
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=seed,
            **umap_cfg,
        ).fit_transform(features)

    raise ValueError(f"Unsupported reduction method: {cfg.reduction.method!r}; expected pca, tsne, or umap")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _run_output_dir(cfg: DictConfig) -> Path:
    method = str(cfg.reduction.method).lower()
    name = "__".join(
        [
            _safe_component(Path(cfg.video_dir).expanduser().name),
            _safe_component(str(cfg.backbone.name)),
            _safe_component(str(cfg.representation)),
            _safe_component(method),
        ]
    )
    output_dir = Path(cfg.output_dir).expanduser()
    return output_dir / name


def _extract_features(
    dataset: MotionFrameDataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches = []
    frame_indices: list[int] = []
    image_paths: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = model(batch["image"].to(device, non_blocking=True))
            if features.ndim != 2:
                raise ValueError(
                    "Selected representation did not produce one vector per frame; "
                    f"got feature shape {tuple(features.shape)}"
                )
            feature_batches.append(features.detach().cpu())
            frame_indices.extend(int(index) for index in batch["frame_index"])
            image_paths.extend(batch["image_path"])

    if not feature_batches:
        raise ValueError("No frame features were extracted")
    return torch.cat(feature_batches, dim=0).numpy(), frame_indices, image_paths


def _write_outputs(
    output_dir: Path,
    coords: np.ndarray,
    frame_indices: list[int],
    image_paths: list[str],
    cfg: DictConfig,
    model: Any,
    image_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "image_path", "component_1", "component_2"],
        )
        writer.writeheader()
        for frame_index, image_path, point in zip(frame_indices, image_paths, coords):
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "image_path": image_path,
                    "component_1": float(point[0]),
                    "component_2": float(point[1]),
                }
            )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    if len(coords) > 1:
        ax.plot(coords[:, 0], coords[:, 1], color="0.55", linewidth=1.0, alpha=0.7, zorder=1)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=frame_indices,
        cmap="viridis",
        s=30,
        zorder=2,
    )
    ax.scatter(coords[0, 0], coords[0, 1], marker="o", facecolors="none", edgecolors="white", s=100, zorder=3)
    ax.scatter(coords[-1, 0], coords[-1, 1], marker="x", color="white", s=65, zorder=3)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Frame index")
    ax.set_xlabel("Reduced component 1")
    ax.set_ylabel("Reduced component 2")
    ax.set_title(
        f"{Path(cfg.video_dir).expanduser().name} | {cfg.backbone.name} | "
        f"{cfg.representation} | {str(cfg.reduction.method).lower()}"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "representation.png")
    plt.close(fig)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved["resolved_output_dir"] = str(output_dir.resolve())
    resolved["resolved_image_size"] = image_size
    resolved["resolved_backbone_timm_id"] = getattr(model, "timm_id", None)
    resolved["num_frames"] = len(frame_indices)
    OmegaConf.save(OmegaConf.create(resolved), output_dir / "config.yaml", resolve=True)


def run_representation_visualization(cfg: DictConfig) -> Path:
    pool = representation_pool(str(cfg.representation))
    device_name = str(cfg.device).lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model, _ = load_backbone(
        str(cfg.backbone.name),
        pool=pool,
        image_mean=cfg.backbone.get("image_mean"),
        custom_mean=cfg.backbone.get("custom_mean"),
        custom_std=cfg.backbone.get("custom_std"),
        img_size=None if cfg.image_size is None else int(cfg.image_size),
    )
    model = model.to(device)
    image_size = cfg.image_size
    image_size = _model_input_size(model) if image_size is None else int(image_size)
    norm_mean = getattr(model, "normalize_mean", None)
    norm_std = getattr(model, "normalize_std", None)
    if norm_mean is None or norm_std is None:
        raise ValueError("Selected backbone did not expose normalization statistics")

    dataset = MotionFrameDataset(cfg.video_dir, image_size, norm_mean, norm_std)
    features, frame_indices, image_paths = _extract_features(
        dataset,
        model,
        device,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
    )
    coords = reduce_features(features, cfg)
    output_dir = _run_output_dir(cfg)
    _write_outputs(output_dir, coords, frame_indices, image_paths, cfg, model, image_size)
    return output_dir


@hydra.main(config_name="representation_visualization", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    run_representation_visualization(cfg)


if __name__ == "__main__":
    main()
