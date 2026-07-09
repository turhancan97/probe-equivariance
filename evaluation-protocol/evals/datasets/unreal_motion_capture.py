"""Dataset that reads unreal-motion-capture's actual output tree directly.

Expected layout, produced by unreal-motion-capture/main.py + motion.py:

    <root>/<environment>/<object_name>/<mode_name>/
        <mode_name>_line_000.jpg ... (or _orbit_NNN.jpg)
        <mode_name>_camera_poses.json   (camera_line, camera_orbit)
        <mode_name>_object_poses.json   (object_line, object_orbit)
        <mode_name>_meta.json           (obj_loc, mode_cfg)

Regression targets are computed from the real saved poses + meta, not from
synthetic frame-index math:
  - camera_line / object_line -> 1D target, position along Y normalized by
    the recording's configured line_half_span.
  - camera_orbit / object_orbit -> 2D target [cos theta, sin theta], theta
    from the recorded (x, y) relative to the object's spawn location.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

LINE_MODES = {"camera_line", "object_line"}
ORBIT_MODES = {"camera_orbit", "object_orbit"}


@dataclass
class _Sample:
    image_path: Path
    environment: str
    mode: str
    object_class: str
    frame_index: int
    location: Dict[str, float]
    target: Optional[torch.Tensor] = None


def regression_dim(mode: str) -> int:
    return 2 if mode in ORBIT_MODES else 1


class UnrealMotionCaptureTask(Dataset):
    """Samples grouped by (environment, mode, object_class)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        split_ratio: float = 0.90,
        test_ratio: float = 0.00,
        environments: Optional[Sequence[str]] = None,
        modes: Optional[Sequence[str]] = None,
        seed: int = 8,
        name: str = "unreal_motion_capture",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split_ratio = float(split_ratio)
        self.test_ratio = float(test_ratio)
        self.allowed_environments = set(environments) if environments else None
        self.allowed_modes = set(modes) if modes else None
        self.name = name
        self.seed = seed

        if image_mean == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif image_mean == "clip":
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        else:
            mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.samples: List[_Sample] = []
        self._groups: DefaultDict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.group_indices_by_split: Dict[str, DefaultDict[Tuple[str, str, str], List[int]]] = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }
        self.split_indices: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}

        self._index_samples()
        self._assign_targets()
        self._build_splits()
        self._set_active_split(split)

    # ------------------------------------------------------------------
    def _index_samples(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        for environment_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            environment_name = environment_dir.name
            if self.allowed_environments and environment_name not in self.allowed_environments:
                continue

            for object_dir in sorted(p for p in environment_dir.iterdir() if p.is_dir()):
                object_name = object_dir.name

                for mode_dir in sorted(p for p in object_dir.iterdir() if p.is_dir()):
                    mode_name = mode_dir.name
                    if self.allowed_modes and mode_name not in self.allowed_modes:
                        continue
                    if mode_name not in LINE_MODES | ORBIT_MODES:
                        continue

                    self._index_mode_dir(mode_dir, environment_name, mode_name, object_name)

        self.available_environments = sorted({s.environment for s in self.samples})
        self.available_modes = sorted({s.mode for s in self.samples})

    def _index_mode_dir(self, mode_dir: Path, environment: str, mode: str, object_name: str) -> None:
        pose_suffix = "camera_poses.json" if mode.startswith("camera_") else "object_poses.json"
        pose_path = mode_dir / f"{mode}_{pose_suffix}"
        meta_path = mode_dir / f"{mode}_meta.json"
        if not pose_path.exists() or not meta_path.exists():
            return

        with pose_path.open("r") as f:
            poses = json.load(f)
        with meta_path.open("r") as f:
            meta = json.load(f)

        key = (environment, mode, object_name)
        for frame_index, entry in enumerate(poses):
            image_path = mode_dir / entry["frame"]
            if not image_path.exists():
                continue
            sample_index = len(self.samples)
            self.samples.append(
                _Sample(
                    image_path=image_path,
                    environment=environment,
                    mode=mode,
                    object_class=object_name,
                    frame_index=frame_index,
                    location=entry["location"],
                )
            )
            self._groups[key].append(sample_index)

        self._meta_by_group = getattr(self, "_meta_by_group", {})
        self._meta_by_group[key] = meta

    # ------------------------------------------------------------------
    def _assign_targets(self) -> None:
        for key, indices in self._groups.items():
            _, mode, _ = key
            meta = self._meta_by_group[key]
            obj_loc = meta["obj_loc"]
            mode_cfg = meta["mode_cfg"]

            sorted_indices = sorted(indices, key=lambda idx: self.samples[idx].frame_index)

            if mode in LINE_MODES:
                half_span = float(mode_cfg["line_half_span"])
                for idx in sorted_indices:
                    loc = self.samples[idx].location
                    fraction = (loc["y"] - obj_loc["y"] + half_span) / (2.0 * half_span)
                    clipped = max(0.0, min(1.0, fraction))
                    self.samples[idx].target = torch.tensor([clipped], dtype=torch.float32)
            else:  # orbit modes
                for idx in sorted_indices:
                    loc = self.samples[idx].location
                    theta = math.atan2(loc["y"] - obj_loc["y"], loc["x"] - obj_loc["x"])
                    self.samples[idx].target = torch.tensor(
                        [math.cos(theta), math.sin(theta)], dtype=torch.float32
                    )

    # ------------------------------------------------------------------
    def _build_splits(self) -> None:
        rng = np.random.RandomState(self.seed)
        for key, indices in self._groups.items():
            sorted_indices = sorted(indices, key=lambda idx: self.samples[idx].frame_index)
            n = len(sorted_indices)
            if n == 0:
                continue

            train_cut = int(round(self.split_ratio * n))
            train_cut = max(1, min(train_cut, n))
            if train_cut >= n and n > 1:
                train_cut = n - 1

            train_indices = sorted_indices[:train_cut]
            remaining = sorted_indices[train_cut:]

            test_count = int(round(self.test_ratio * n)) if self.test_ratio > 0 else 0
            test_count = min(test_count, len(remaining))
            if test_count > 0:
                remaining = list(remaining)
                rng.shuffle(remaining)
                test_indices = remaining[:test_count]
                val_indices = remaining[test_count:]
            else:
                test_indices = []
                val_indices = remaining

            for idx in train_indices:
                self.split_indices["train"].append(idx)
                self.group_indices_by_split["train"][key].append(idx)
            for idx in val_indices:
                self.split_indices["valid"].append(idx)
                self.group_indices_by_split["valid"][key].append(idx)
            for idx in test_indices:
                self.split_indices["test"].append(idx)
                self.group_indices_by_split["test"][key].append(idx)

        self.split_indices["trainval"] = self.split_indices["train"] + self.split_indices["valid"]

    def _set_active_split(self, split: str) -> None:
        split_key = "valid" if split in {"val", "valid"} else split
        if split_key not in self.split_indices:
            raise ValueError(f"Unsupported split: {split}")
        self.active_split = split_key
        self.active_indices = self.split_indices[split_key]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_idx = self.active_indices[index]
        sample = self.samples[sample_idx]
        with Image.open(sample.image_path).convert("RGB") as image:
            image_t = image.copy()
        image_t = self.transform(image_t)

        return {
            "image": image_t,
            "target": sample.target.clone(),
            "environment": sample.environment,
            "mode": sample.mode,
            "object_class": sample.object_class,
            "frame_index": torch.tensor(sample.frame_index, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def set_split(self, split: str) -> None:
        self._set_active_split(split)

    def groups_for_split(self, split: str) -> Dict[Tuple[str, str, str], List[int]]:
        split_key = "valid" if split in {"val", "valid"} else split
        return self.group_indices_by_split.get(split_key, {})
