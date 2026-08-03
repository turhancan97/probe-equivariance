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
from evals.models.backbone import BACKBONE_REGISTRY, FrozenBackbone
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


class _DummyTokenBackbone(nn.Module):
    def __init__(self, feat_dim: int = 8, num_tokens: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tokens = num_tokens
        self.checkpoint_name = "dummy_clip_patch"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        if pooled.shape[1] < self.feat_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.feat_dim - pooled.shape[1]))
        pooled = pooled[:, : self.feat_dim]
        offsets = torch.linspace(0.0, 0.3, steps=self.num_tokens, device=pooled.device)
        return torch.stack([pooled + offset for offset in offsets], dim=1)


class _DummyPatchTimmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.num_prefix_tokens = 1

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        tokens = torch.tensor(
            [
                [-9.0, -9.0, -9.0, -9.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=images.dtype,
            device=images.device,
        )
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)


class _DummyRegistryTimmModel(nn.Module):
    def __init__(self, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6)):
        super().__init__()
        self.num_features = 4
        self.num_prefix_tokens = 1
        self.pretrained_cfg = {"mean": mean, "std": std}

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.zeros(batch_size, 2, self.num_features, dtype=images.dtype, device=images.device)


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

    def test_patch_pool_excludes_prefix_tokens(self):
        with mock.patch(
            "evals.models.backbone._create_model_with_fallbacks",
            return_value=(_DummyPatchTimmModel(), "dummy_patch_model"),
        ):
            model = FrozenBackbone("dummy_patch_model", pool="patch")
            feats = model(torch.zeros(2, 3, 32, 32))

        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            ]
        )
        self.assertTrue(torch.equal(feats.cpu(), expected))

    def test_backbone_registry_resolves_all_known_ids(self):
        for friendly_name, expected_timm_id in BACKBONE_REGISTRY.items():
            with self.subTest(backbone=friendly_name):
                with mock.patch(
                    "evals.models.backbone._create_model_with_fallbacks",
                    return_value=(_DummyRegistryTimmModel(), expected_timm_id),
                ) as mocked_create:
                    model = FrozenBackbone(friendly_name, pool="mean")

                mocked_create.assert_called_once_with(expected_timm_id)
                self.assertEqual(model.timm_id, expected_timm_id)
                self.assertEqual(model.feat_dim, 4)
                self.assertEqual(model.normalize_mean, [0.1, 0.2, 0.3])
                self.assertEqual(model.normalize_std, [0.4, 0.5, 0.6])

    def test_run_equivariance_and_visualization_smoke(self):
        cfg = self._build_train_cfg(
            experiment_name="smoke_multi_mode",
            backbone={"name": "clip_b16_laion", "pool": "mean"},
            probe={"_target_": "evals.models.probe.RegressionHead", "use_layernorm": True},
        )
        vis_cfg = self._build_vis_cfg("smoke_multi_mode")

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

    def test_run_equivariance_and_visualization_efficient_probing_smoke(self):
        cfg = self._build_train_cfg(
            experiment_name="smoke_efficient_probing",
            backbone={"name": "clip_b16_laion", "pool": "patch"},
            probe={
                "_target_": "evals.models.probe.EfficientProbingHead",
                "num_queries": 2,
                "num_heads": 1,
                "d_out": 1,
                "use_layernorm": True,
                "qkv_bias": False,
                "qk_scale": None,
            },
        )
        vis_cfg = self._build_vis_cfg("smoke_efficient_probing")

        with mock.patch("train_equivariance.load_backbone", return_value=(_DummyTokenBackbone(), 8)):
            run_equivariance(cfg)
        with mock.patch("visualize_equivariance.load_backbone", return_value=(_DummyTokenBackbone(), 8)):
            run_visualization(vis_cfg)

        result_dir = self.output_root / "equivariance_smoke_efficient_probing"
        object_metrics = (result_dir / "object_metrics.csv").read_text()
        checkpoint_manifest = json.loads((result_dir / "checkpoint_manifest.json").read_text())
        run_config = (result_dir / "run_config.yaml").read_text()

        self.assertIn("camera_line", object_metrics)
        self.assertIn("camera_orbit", object_metrics)
        self.assertEqual(checkpoint_manifest["head"], "evals.models.probe.EfficientProbingHead")
        self.assertIn("pool: patch", run_config)
        self.assertIn("EfficientProbingHead", run_config)

        for mode in ("camera_line", "camera_orbit"):
            plot_path = result_dir / "visualizations" / f"env_a__{mode}__cube.png"
            self.assertTrue(plot_path.exists(), str(plot_path))

    def test_efficient_probing_requires_patch_pool(self):
        cfg = self._build_train_cfg(
            experiment_name="invalid_ep_pool",
            backbone={"name": "clip_b16_laion", "pool": "mean"},
            probe={
                "_target_": "evals.models.probe.EfficientProbingHead",
                "num_queries": 2,
                "num_heads": 1,
                "d_out": 1,
                "use_layernorm": True,
                "qkv_bias": False,
                "qk_scale": None,
            },
        )

        with self.assertRaisesRegex(ValueError, "requires backbone.pool='patch'"):
            run_equivariance(cfg)

    def test_efficient_probing_rejects_multi_head_v1(self):
        cfg = self._build_train_cfg(
            experiment_name="invalid_ep_heads",
            backbone={"name": "clip_b16_laion", "pool": "patch"},
            probe={
                "_target_": "evals.models.probe.EfficientProbingHead",
                "num_queries": 2,
                "num_heads": 2,
                "d_out": 1,
                "use_layernorm": True,
                "qkv_bias": False,
                "qk_scale": None,
            },
        )

        with mock.patch("train_equivariance.load_backbone", return_value=(_DummyTokenBackbone(), 8)):
            with self.assertRaisesRegex(ValueError, "num_heads=1"):
                run_equivariance(cfg)

    def _build_train_cfg(self, experiment_name: str, backbone: dict, probe: dict):
        return OmegaConf.create(
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
                "backbone": backbone,
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
                "probe": probe,
                "experiment_name": experiment_name,
                "experiment_model": backbone["name"],
                "training": {
                    "min_samples_per_object": 1,
                    "eval_every_epochs": 1,
                    "save_checkpoints": True,
                },
                "output_dir": str(self.output_root),
            }
        )

    def _build_vis_cfg(self, experiment_name: str):
        return OmegaConf.create(
            {
                "result_dir": str(self.output_root / f"equivariance_{experiment_name}"),
                "run_config_path": None,
                "dataset_root": None,
                "output_dir": None,
                "output_subdir": "visualizations",
                "checkpoint_subdir": "checkpoints",
                "splits": ["train", "valid", "test"],
            }
        )

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
