import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualize_representations import (  # noqa: E402
    MotionFrameDataset,
    reduce_features,
    representation_pool,
    run_representation_visualization,
)


class _DummyTimmMetadata(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_cfg = {"input_size": (3, 16, 16)}


class _DummyRepresentationBackbone(nn.Module):
    def __init__(self, pool: str):
        super().__init__()
        self.pool = pool
        self.feat_dim = 4
        self.checkpoint_name = "dummy_representation"
        self.timm_id = "dummy_representation"
        self.normalize_mean = [0.0, 0.0, 0.0]
        self.normalize_std = [1.0, 1.0, 1.0]
        self.model = _DummyTimmMetadata()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.mean(dim=(2, 3))
        return torch.cat([values, values[:, :1]], dim=1)


class RepresentationVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="representation-visualization-test-"))
        self.video_dir = self.temp_dir / "camera_line"
        self.video_dir.mkdir()
        self._write_video()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_video(self):
        frame_names = ["z_frame.jpg", "a_frame.jpg", "m_frame.jpg", "b_frame.jpg"]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for frame_name, color in zip(frame_names, colors):
            Image.new("RGB", (12, 12), color=color).save(self.video_dir / frame_name)
        poses = [{"frame": name} for name in [frame_names[2], frame_names[0], frame_names[3], frame_names[1]]]
        (self.video_dir / "camera_line_camera_poses.json").write_text(json.dumps(poses), encoding="utf-8")

    def _config(self, representation="cls", method="pca"):
        return OmegaConf.create(
            {
                "video_dir": str(self.video_dir),
                "output_dir": str(self.temp_dir / "outputs"),
                "image_size": 16,
                "representation": representation,
                "batch_size": 2,
                "num_workers": 0,
                "device": "cpu",
                "random_seed": 8,
                "backbone": {
                    "name": "dummy_representation",
                    "image_mean": "custom",
                    "custom_mean": [0.0, 0.0, 0.0],
                    "custom_std": [1.0, 1.0, 1.0],
                },
                "reduction": {
                    "method": method,
                    "pca": {"whiten": False, "svd_solver": "auto"},
                    "tsne": {
                        "perplexity": None,
                        "learning_rate": "auto",
                        "init": "pca",
                        "max_iter": 250,
                    },
                    "umap": {"n_neighbors": None, "min_dist": 0.1, "metric": "euclidean"},
                },
            }
        )

    def test_pose_json_order_and_outputs(self):
        cfg = self._config()
        with mock.patch(
            "visualize_representations.load_backbone",
            return_value=(_DummyRepresentationBackbone("cls"), 4),
        ) as load_mock:
            output_dir = run_representation_visualization(cfg)

        load_mock.assert_called_once()
        self.assertEqual(load_mock.call_args.kwargs["pool"], "cls")
        self.assertTrue((output_dir / "representation.png").exists())
        self.assertTrue((output_dir / "config.yaml").exists())

        with (output_dir / "frames.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["frame_index"]) for row in rows], [0, 1, 2, 3])
        self.assertEqual(
            [Path(row["image_path"]).name for row in rows],
            ["m_frame.jpg", "z_frame.jpg", "b_frame.jpg", "a_frame.jpg"],
        )
        self.assertTrue(all(row["component_1"] and row["component_2"] for row in rows))

    def test_representation_mapping_and_patch_mean(self):
        self.assertEqual(representation_pool("cls"), "cls")
        self.assertEqual(representation_pool("patch_mean"), "mean")
        with self.assertRaises(ValueError):
            representation_pool("patch")

        cfg = self._config(representation="patch_mean")
        with mock.patch(
            "visualize_representations.load_backbone",
            return_value=(_DummyRepresentationBackbone("mean"), 4),
        ) as load_mock:
            run_representation_visualization(cfg)
        self.assertEqual(load_mock.call_args.kwargs["pool"], "mean")
        self.assertEqual(load_mock.call_args.kwargs["img_size"], 16)

    def test_invalid_pose_metadata_is_rejected(self):
        broken_dir = self.temp_dir / "broken"
        broken_dir.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "pose JSON"):
            MotionFrameDataset(broken_dir, 16, [0.0] * 3, [1.0] * 3)

    def test_seeded_tsne_is_repeatable(self):
        cfg = self._config(method="tsne")
        features = np.arange(32, dtype=np.float32).reshape(8, 4)
        first = reduce_features(features, cfg)
        second = reduce_features(features, cfg)
        self.assertEqual(first.shape, (8, 2))
        np.testing.assert_allclose(first, second)

    def test_umap_reports_optional_dependency(self):
        cfg = self._config(method="umap")
        features = np.arange(16, dtype=np.float32).reshape(4, 4)
        with mock.patch.dict("sys.modules", {"umap": None}):
            with self.assertRaisesRegex(ModuleNotFoundError, "umap-learn"):
                reduce_features(features, cfg)


if __name__ == "__main__":
    unittest.main()
