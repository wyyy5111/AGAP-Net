"""Train AGAP-Net with one speaker or session group held out at a time."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .data import FeatureStore
from .engine import choose_device, read_config, train_fold


def group_directory(group):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", group).strip("._") or "group"
    return slug + "_" + hashlib.sha256(group.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--held-out", help="Train one fold; otherwise evaluate every group.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model_config, config = read_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    store = FeatureStore(args.cache)
    groups = [args.held_out] if args.held_out else store.groups()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory for this run.")
    rows = []
    for group in groups:
        print(f"Held-out group: {group}", flush=True)
        fold_output = output / group_directory(group)
        rows.append(train_fold(store, group, fold_output, model_config,
                               config, choose_device(args.device)))
        print(f"Checkpoint: {fold_output / 'model.pt'}", flush=True)
    aggregate = {}
    for metric in ["WA", "UAR", "macroF1"]:
        values = np.asarray([r["metrics"][metric] for r in rows]) * 100
        aggregate[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    (output / "summary.json").write_text(json.dumps({"folds": rows, "fold_mean_std_percent": aggregate}, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
