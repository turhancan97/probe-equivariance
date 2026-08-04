#!/usr/bin/env python3
"""Create ranked backbone comparison plots from an aggregate metrics CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "val_rmse": "Validation RMSE",
    "test_rmse": "Test RMSE",
}
SIZE_ORDER = ("small", "base", "large")
FAMILY_PREFIXES = (
    "perception_encoder",
    "supervised",
    "dinov3",
    "dinov2",
    "deit3",
    "siglip",
    "clip",
    "mae",
    "dino",
)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _backbone_family(backbone: str) -> str:
    normalized = backbone.lower().replace("-", "_")
    for prefix in FAMILY_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}_"):
            return prefix
    return normalized.split("_")[0]


def _backbone_size(backbone: str) -> str:
    normalized = backbone.lower().replace("-", "_")
    if (
        "small" in normalized
        or re.search(r"(?:^|_)vit?s\d", normalized)
        or re.search(r"(?:^|_)s\d+(?:_|$)", normalized)
    ):
        return "small"
    if (
        "large" in normalized
        or re.search(r"(?:^|_)vitl\d", normalized)
        or re.search(r"(?:^|_)l\d+(?:_|$)", normalized)
    ):
        return "large"
    if (
        "base" in normalized
        or re.search(r"(?:^|_)vitb\d", normalized)
        or re.search(r"(?:^|_)b\d+(?:_|$)", normalized)
    ):
        return "base"
    return "unknown"


def load_metrics(metrics_csv: str | Path) -> list[dict[str, str]]:
    metrics_csv = Path(metrics_csv).expanduser()
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Combined metrics CSV does not exist: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"environment", "mode", "backbone", "pool", *METRIC_LABELS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing expected columns in {metrics_csv}: {sorted(missing)}")
        return list(reader)


def _group_values(
    rows: list[dict[str, str]],
    metric_names: list[str],
) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for metric in metric_names:
            values[row["backbone"]][metric].append(_finite_float(row[metric]))
    return values


def _rank_backbones(
    backbone_values: dict[str, dict[str, list[float]]],
    order_by: str,
) -> list[str]:
    fallback = "test_rmse" if order_by == "val_rmse" else "val_rmse"

    def rank_key(backbone: str) -> tuple[float, float, str]:
        primary = _mean(backbone_values[backbone].get(order_by, []))
        secondary = _mean(backbone_values[backbone].get(fallback, []))
        return (
            primary if math.isfinite(primary) else float("inf"),
            secondary if math.isfinite(secondary) else float("inf"),
            backbone,
        )

    return sorted(backbone_values, key=rank_key)


def _draw_bars(
    ax: plt.Axes,
    backbone_values: dict[str, dict[str, list[float]]],
    backbones: list[str],
    metric_names: list[str],
) -> None:
    x = np.arange(len(backbones), dtype=float)
    width = 0.8 / len(metric_names)
    for metric_index, metric in enumerate(metric_names):
        values = [_mean(backbone_values[backbone].get(metric, [])) for backbone in backbones]
        offsets = x + (metric_index - (len(metric_names) - 1) / 2.0) * width
        ax.bar(offsets, values, width=width, label=METRIC_LABELS[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(backbones, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)


def _plot_all_backbones(
    rows: list[dict[str, str]],
    output_path: Path,
    environment: str,
    mode: str,
    pool: str,
    metric_names: list[str],
    order_by: str,
) -> None:
    backbone_values = _group_values(rows, metric_names)
    backbones = _rank_backbones(backbone_values, order_by)
    fig_width = max(8.0, 1.2 * len(backbones))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), dpi=160)
    _draw_bars(ax, backbone_values, backbones, metric_names)
    ax.set_title(f"Backbone comparison | {environment} | {mode} | pool: {pool}")
    ax.set_xlabel("Backbone (best to worst)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_by_size(
    rows: list[dict[str, str]],
    output_path: Path,
    environment: str,
    mode: str,
    pool: str,
    metric_names: list[str],
    order_by: str,
) -> None:
    grouped = {
        size: _group_values(
            [row for row in rows if _backbone_size(row["backbone"]) == size],
            metric_names,
        )
        for size in SIZE_ORDER
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160, squeeze=False)
    axes = axes[0]
    for ax, size in zip(axes, SIZE_ORDER):
        backbone_values = grouped[size]
        if backbone_values:
            backbones = _rank_backbones(backbone_values, order_by)
            _draw_bars(ax, backbone_values, backbones, metric_names)
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No backbones", ha="center", va="center")
            ax.set_xticks([])
        ax.set_title(size.capitalize())
        ax.set_xlabel("Backbone (best to worst)")
        ax.set_ylabel("RMSE")
    fig.suptitle(f"Backbone comparison by size | {environment} | {mode} | pool: {pool}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_by_family(
    rows: list[dict[str, str]],
    output_path: Path,
    environment: str,
    mode: str,
    pool: str,
    metric_names: list[str],
    order_by: str,
) -> None:
    family_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        family_rows[_backbone_family(row["backbone"])].append(row)

    families = sorted(family_rows)
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(max(8.0, 5.5 * len(families)), 5.5),
        dpi=160,
        squeeze=False,
    )
    axes = axes[0]
    for ax, family in zip(axes, families):
        backbone_values = _group_values(family_rows[family], metric_names)
        backbones = _rank_backbones(backbone_values, order_by)
        _draw_bars(ax, backbone_values, backbones, metric_names)
        ax.set_title(family)
        ax.set_xlabel("Backbone (best to worst)")
        ax.set_ylabel("RMSE")
        ax.legend(loc="best")
    fig.suptitle(f"Backbone comparison by family | {environment} | {mode} | pool: {pool}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_backbone_metrics(
    metrics_csv: str | Path,
    output_dir: str | Path,
    metric_names: list[str] | None = None,
    order_by: str = "val_rmse",
) -> list[Path]:
    metric_names = metric_names or list(METRIC_LABELS)
    if not metric_names:
        raise ValueError("At least one metric must be selected")
    invalid = [metric for metric in metric_names if metric not in METRIC_LABELS]
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}; expected one of {sorted(METRIC_LABELS)}")
    if order_by not in METRIC_LABELS:
        raise ValueError(f"Unsupported order metric: {order_by}; expected one of {sorted(METRIC_LABELS)}")

    rows = load_metrics(metrics_csv)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["mode"], row["pool"])].append(row)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for (environment, mode, pool), group_rows in sorted(grouped.items()):
        stem = (
            f"{_safe_component(environment)}__{_safe_component(mode)}__"
            f"{_safe_component(pool)}"
        )
        all_path = output_dir / f"{stem}.png"
        size_path = output_dir / f"{stem}__by_size.png"
        family_path = output_dir / f"{stem}__by_family.png"
        _plot_all_backbones(
            group_rows, all_path, environment, mode, pool, metric_names, order_by
        )
        _plot_by_size(
            group_rows, size_path, environment, mode, pool, metric_names, order_by
        )
        _plot_by_family(
            group_rows, family_path, environment, mode, pool, metric_names, order_by
        )
        output_paths.extend((all_path, size_path, family_path))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        default="results/combined_backbone_metrics.csv",
        help="CSV created by aggregate_backbone_metrics.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/backbone_comparisons",
        help="Directory for ranked and grouped plots",
    )
    parser.add_argument(
        "--metrics",
        default="val_rmse,test_rmse",
        help="Comma-separated metrics to plot; supported values: val_rmse,test_rmse",
    )
    parser.add_argument(
        "--order-by",
        choices=sorted(METRIC_LABELS),
        default="val_rmse",
        help="Primary ascending metric used to rank backbones; the other metric breaks ties",
    )
    args = parser.parse_args()
    metric_names = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    output_paths = plot_backbone_metrics(
        args.metrics_csv,
        args.output_dir,
        metric_names,
        order_by=args.order_by,
    )
    print(f"Wrote {len(output_paths)} backbone comparison plots to {Path(args.output_dir).expanduser()}")


if __name__ == "__main__":
    main()
