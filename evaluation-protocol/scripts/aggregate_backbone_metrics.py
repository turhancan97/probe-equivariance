#!/usr/bin/env python3
"""Aggregate per-run object metrics into one backbone-comparison CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


OUTPUT_FIELDS = [
    "result_dir",
    "timestamp",
    "seed",
    "experiment",
    "environment",
    "mode",
    "object",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "num_train",
    "num_val",
    "num_test",
    "backbone",
    "pool",
    "head",
]


def _timestamp_key(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%d%m%Y-%H%M")
    except (TypeError, ValueError):
        return datetime.min


def _load_run_metadata(result_dir: Path) -> dict[str, str]:
    config_path = result_dir / "run_config.yaml"
    if not config_path.exists():
        return {}
    config = OmegaConf.load(config_path)
    system_cfg = config.get("system", {})
    backbone_cfg = config.get("backbone", {})
    probe_cfg = config.get("probe", {})
    return {
        "seed": str(system_cfg.get("random_seed", "")),
        "backbone": str(backbone_cfg.get("name", "")),
        "pool": str(backbone_cfg.get("pool", "")),
        "head": str(probe_cfg.get("_target_", "")),
    }


def _normalize_row(
    row: dict[str, str], result_dir: Path, metadata: dict[str, str]
) -> dict[str, str]:
    return {
        "result_dir": str(result_dir),
        "timestamp": row.get("Timestamp", ""),
        "seed": row.get("Seed", "") or metadata.get("seed", ""),
        "experiment": row.get("Experiment", ""),
        "environment": row.get("Environment", ""),
        "mode": row.get("Mode", ""),
        "object": row.get("Object", ""),
        "train_rmse": row.get("Train RMSE", ""),
        "val_rmse": row.get("Val RMSE", ""),
        "test_rmse": row.get("Test RMSE", ""),
        "num_train": row.get("Num Train", ""),
        "num_val": row.get("Num Val", ""),
        "num_test": row.get("Num Test", ""),
        "backbone": row.get("Backbone", "") or metadata.get("backbone", ""),
        "pool": row.get("Pool", "") or metadata.get("pool", ""),
        "head": row.get("Head", "") or metadata.get("head", ""),
    }


def aggregate_metrics(
    results_root: str | Path,
    output_path: str | Path,
    keep_history: bool = False,
) -> list[dict[str, str]]:
    results_root = Path(results_root).expanduser()
    output_path = Path(output_path).expanduser()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    input_paths = sorted(results_root.glob("equivariance_*/object_metrics.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No equivariance object_metrics.csv files found under: {results_root}")

    rows: list[dict[str, str]] = []
    for input_path in input_paths:
        metadata = _load_run_metadata(input_path.parent)
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            source_required = {
                "timestamp",
                "experiment",
                "environment",
                "mode",
                "object",
                "train_rmse",
                "val_rmse",
                "test_rmse",
                "num_train",
                "num_val",
                "num_test",
            }
            missing = source_required - {
                {
                    "Timestamp": "timestamp",
                    "Experiment": "experiment",
                    "Environment": "environment",
                    "Mode": "mode",
                    "Object": "object",
                    "Train RMSE": "train_rmse",
                    "Val RMSE": "val_rmse",
                    "Test RMSE": "test_rmse",
                    "Num Train": "num_train",
                    "Num Val": "num_val",
                    "Num Test": "num_test",
                }.get(field, field)
                for field in (reader.fieldnames or [])
            }
            if missing:
                raise ValueError(
                    f"Missing expected columns in {input_path}: {sorted(missing)}"
                )
            for row in reader:
                normalized = _normalize_row(row, input_path.parent, metadata)
                if not normalized["backbone"]:
                    raise ValueError(
                        f"Missing Backbone value in {input_path} and run_config.yaml"
                    )
                if not normalized["pool"]:
                    raise ValueError(
                        f"Missing Pool value in {input_path} and run_config.yaml"
                    )
                if not normalized["seed"]:
                    raise ValueError(
                        f"Missing random seed in {input_path.parent / 'run_config.yaml'}"
                    )
                rows.append(normalized)

    if not keep_history:
        latest: dict[tuple[str, ...], tuple[datetime, int, dict[str, str]]] = {}
        for order, row in enumerate(rows):
            key = (
                row["result_dir"],
                row["experiment"],
                row["environment"],
                row["mode"],
                row["object"],
                row["backbone"],
                row["pool"],
                row["seed"],
            )
            candidate = (_timestamp_key(row["timestamp"]), order, row)
            if key not in latest or candidate[:2] >= latest[key][:2]:
                latest[key] = candidate
        rows = [candidate[2] for candidate in latest.values()]

    rows.sort(
        key=lambda row: (
            row["environment"],
            row["mode"],
            row["pool"],
            row["backbone"],
            row["seed"],
            row["object"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Root containing equivariance_* result directories")
    parser.add_argument(
        "--output",
        default="results/combined_backbone_metrics.csv",
        help="Combined CSV output path",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="Keep repeated historical rows instead of retaining the latest row per result/object key",
    )
    args = parser.parse_args()
    rows = aggregate_metrics(args.results_root, args.output, keep_history=args.keep_history)
    print(f"Wrote {len(rows)} rows to {Path(args.output).expanduser()}")


if __name__ == "__main__":
    main()
