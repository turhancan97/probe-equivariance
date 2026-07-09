import importlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.datasets.builder import build_loader
from train_equivariance import run_equivariance
from visualize_equivariance import run_visualization


class _DummyBackbone(nn.Module):
    def __init__(self, feat_dim: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.checkpoint_name = "dummy_clip"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        if pooled.shape[1] < self.feat_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.feat_dim - pooled.shape[1]))
        return pooled[:, : self.feat_dim]


class EvaluationProtocolSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="probe-equivariance-test-")
        self.dataset_root = Path(self.temp_dir) / "dataset"
        self.output_root = Path(self.temp_dir) / "results"
        self._build_dataset(self.dataset_root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_train_module_imports_without_wandb(self):
        module = importlib.import_module("train_equivariance")
        self.assertTrue(hasattr(module, "run_equivariance"))

    def test_build_loader_batches_single_mode_targets(self):
        dataset_cfg = OmegaConf.create(
            {
                "_target_": "evals.datasets.unreal_motion_capture.UnrealMotionCaptureTask",
                "root": str(self.dataset_root),
                "image_size": 32,
                "image_mean": "imagenet",
                "split_ratio": 0.5,
                "test_ratio": 0.25,
                "environments": None,
                "modes": None,
                "seed": 8,
                "name": "unreal_motion_capture",
            }
        )
        loader = build_loader(dataset_cfg, "train", batch_size=4, num_workers=0)

        seen_modes = set()
        for batch in loader:
            modes = set(batch["mode"])
            self.assertEqual(len(modes), 1)
            mode = next(iter(modes))
            seen_modes.add(mode)
            expected_dim = 1 if mode.endswith("line") else 2
            self.assertEqual(batch["target"].shape[1], expected_dim)

        self.assertEqual(seen_modes, {"camera_line", "camera_orbit"})

    def test_run_equivariance_and_visualization_smoke(self):
        cfg = OmegaConf.create(
            {
                "system": {"random_seed": 8},
                "note": "",
                "batch_size": 4,
                "wandb": {"use": False},
                "optimizer": {
                    "probe_lr": 0.001,
                    "weight_decay": 0.001,
                    "n_epochs": 1,
                    "warmup_epochs": 0,
                },
                "backbone": {"name": "clip_b16_laion", "pool": "mean"},
                "dataset": {
                    "_target_": "evals.datasets.unreal_motion_capture.UnrealMotionCaptureTask",
                    "root": str(self.dataset_root),
                    "image_size": 32,
                    "image_mean": "imagenet",
                    "split_ratio": 0.5,
                    "test_ratio": 0.25,
                    "environments": None,
                    "modes": None,
                    "seed": 8,
                    "name": "unreal_motion_capture",
                },
                "probe": {
                    "_target_": "evals.models.probe.RegressionHead",
                    "use_layernorm": True,
                },
                "experiment_name": "smoke_multi_mode",
                "experiment_model": "clip_b16_laion",
                "training": {
                    "min_samples_per_object": 1,
                    "eval_every_epochs": 1,
                    "save_checkpoints": True,
                },
                "output_dir": str(self.output_root),
            }
        )

        vis_cfg = OmegaConf.create(
            {
                "result_dir": str(self.output_root / "equivariance_smoke_multi_mode"),
                "run_config_path": None,
                "dataset_root": None,
                "output_dir": None,
                "output_subdir": "visualizations",
                "checkpoint_subdir": "checkpoints",
                "splits": ["train", "valid", "test"],
            }
        )

        with mock.patch("train_equivariance.load_backbone", return_value=(_DummyBackbone(), 8)):
            run_equivariance(cfg)
        with mock.patch("visualize_equivariance.load_backbone", return_value=(_DummyBackbone(), 8)):
            run_visualization(vis_cfg)

        result_dir = self.output_root / "equivariance_smoke_multi_mode"
        object_metrics = (result_dir / "object_metrics.csv").read_text()
        mode_summary = (result_dir / "mode_summary.csv").read_text()
        checkpoint_manifest = json.loads((result_dir / "checkpoint_manifest.json").read_text())
        run_config = (result_dir / "run_config.yaml").read_text()

        self.assertIn("camera_line", object_metrics)
        self.assertIn("camera_orbit", object_metrics)
        self.assertIn("Train RMSE camera_line", mode_summary)
        self.assertIn("Train RMSE camera_orbit", mode_summary)
        self.assertTrue(checkpoint_manifest["save_checkpoints"])
        self.assertIn("save_checkpoints: true", run_config)

        for mode in ("camera_line", "camera_orbit"):
            plot_path = result_dir / "visualizations" / f"env_a__{mode}__cube.png"
            self.assertTrue(plot_path.exists(), str(plot_path))

    def _build_dataset(self, root: Path) -> None:
        obj_loc = {"x": 0.0, "y": 0.0, "z": 0.0}
        modes = {
            "camera_line": {"line_half_span": 10.0},
            "camera_orbit": {"radius": 8.0},
        }

        for mode, mode_cfg in modes.items():
            mode_dir = root / "env_a" / "cube" / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            poses = []

            for frame_idx in range(12):
                if mode == "camera_line":
                    frac = frame_idx / 11
                    location = {"x": -5.0, "y": -10.0 + 20.0 * frac, "z": 3.0}
                    frame_name = f"{mode}_line_{frame_idx:03}.jpg"
                    color = (int(255 * frac), 32, 128)
                else:
                    theta = (frame_idx / 11) * 2.0 * math.pi
                    location = {
                        "x": 8.0 * math.cos(theta),
                        "y": 8.0 * math.sin(theta),
                        "z": 4.0,
                    }
                    frame_name = f"{mode}_orbit_{frame_idx:03}.jpg"
                    color = (64, int(255 * (frame_idx / 11)), 192)

                Image.new("RGB", (32, 32), color=color).save(mode_dir / frame_name)
                poses.append({"frame": frame_name, "location": location})

            pose_file = mode_dir / f"{mode}_camera_poses.json"
            meta_file = mode_dir / f"{mode}_meta.json"
            pose_file.write_text(json.dumps(poses))
            meta_file.write_text(json.dumps({"obj_loc": obj_loc, "mode_cfg": mode_cfg}))


if __name__ == "__main__":
    unittest.main()
