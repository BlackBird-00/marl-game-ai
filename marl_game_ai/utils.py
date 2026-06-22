from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_metrics_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_training(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [row["episode"] for row in rows]
    returns = [row["return"] for row in rows]
    successes = [row["success"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(episodes, returns, color="#2563eb", linewidth=1.8)
    axes[0].set_ylabel("Episode return")
    axes[0].grid(alpha=0.25)
    window = min(25, max(1, len(successes)))
    smoothed = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        smoothed.append(sum(successes[start : i + 1]) / (i - start + 1))
    axes[1].plot(episodes, smoothed, color="#16a34a", linewidth=1.8)
    axes[1].set_ylabel("Success rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
