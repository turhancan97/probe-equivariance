import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_backbone_metrics import aggregate_metrics  # noqa: E402
from plot_backbone_metrics import (  # noqa: E402
    _group_values,
    _metric_stats,
    _rank_backbones,
    plot_backbone_metrics,
)


class MetricsAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="metrics-aggregation-test-"))
        self.results_root = self.temp_dir / "results"
        self.results_root.mkdir()
        self._write_result("parallel_clip_base", "clip_b16_laion", "01082026-1000", 0.2)
        self._write_result(
            "parallel_clip_base_seed_2",
            "clip_b16_laion",
            "01082026-1001",
            0.4,
            seed=2,
        )
        self._write_result("parallel_clip_large", "clip_large_laion", "01082026-1002", 0.1)
        self._write_result("parallel_dinov2_small", "dinov2_vits14", "01082026-1003", 0.3)
        self._write_result(
            "parallel_clip_base_patch",
            "clip_b16_laion",
            "01082026-1004",
            0.4,
            pool="patch",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_result(
        self,
        run_name: str,
        backbone: str,
        timestamp: str,
        val_rmse: float,
        pool: str = "mean",
        seed: int = 1,
    ):
        result_dir = self.results_root / f"equivariance_{run_name}"
        result_dir.mkdir()
        (result_dir / "run_config.yaml").write_text(
            f"system:\n  random_seed: {seed}\n", encoding="utf-8"
        )
        with (result_dir / "object_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Timestamp", "Experiment", "Environment", "Mode", "Object",
                    "Train RMSE", "Val RMSE", "Test RMSE", "Num Train", "Num Val",
                    "Num Test", "Backbone", "Pool", "Head",
                ]
            )
            writer.writerow(
                [
                    timestamp, run_name, "FirstPersonMap", "camera_line", "cube",
                    "0.05", str(val_rmse), "nan", "10", "2", "0", backbone,
                    pool, "RegressionHead",
                ]
            )

    def test_aggregate_and_plot(self):
        combined_csv = self.temp_dir / "combined.csv"
        rows = aggregate_metrics(self.results_root, combined_csv)
        self.assertEqual(len(rows), 5)
        with combined_csv.open(newline="", encoding="utf-8") as handle:
            combined_rows = list(csv.DictReader(handle))
        self.assertEqual(
            {row["backbone"] for row in combined_rows},
            {"clip_b16_laion", "clip_large_laion", "dinov2_vits14"},
        )

        mean_rows = [row for row in combined_rows if row["pool"] == "mean"]
        values = _group_values(mean_rows, ["val_rmse", "test_rmse"])
        self.assertEqual(_rank_backbones(values, "val_rmse")[0], "clip_large_laion")
        mean, std = _metric_stats(values, "clip_b16_laion", "val_rmse")
        self.assertAlmostEqual(mean, 0.3)
        self.assertAlmostEqual(std, 0.1414213562, places=6)

        output_dir = self.temp_dir / "plots"
        plots = plot_backbone_metrics(combined_csv, output_dir)
        self.assertEqual(len(plots), 6)
        self.assertEqual(
            {path.name for path in plots},
            {
                "FirstPersonMap__camera_line__mean.png",
                "FirstPersonMap__camera_line__mean__by_size.png",
                "FirstPersonMap__camera_line__mean__by_family.png",
                "FirstPersonMap__camera_line__patch.png",
                "FirstPersonMap__camera_line__patch__by_size.png",
                "FirstPersonMap__camera_line__patch__by_family.png",
            },
        )
        self.assertTrue(all(path.exists() for path in plots))


if __name__ == "__main__":
    unittest.main()
