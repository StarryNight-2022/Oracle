
"""
Uniformly subsample a JSONL dataset into train/test splits and save them.

Defaults:
- Input: Runtime/GSM8k/grade-school-math/grade_school_math/data/train.jsonl
- Output dir for splits: Runtime/GSM8k/outputs-qwen3-4B/splits
- Ratio: 0.2 for test (80/20 split)

Usage examples:
  # Default 20% split
  python Runtime/scripts/subsample_dataset.py

  # Custom ratio 10% test
  python Runtime/scripts/subsample_dataset.py --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def uniform_sample_indices(n: int, k: int) -> List[int]:
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    step = n / k
    idxs = []
    for i in range(k):
        pos = int(round(i * step))
        if pos >= n:
            pos = n - 1
        idxs.append(pos)
    seen = set()
    uniq = []
    for x in idxs:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return sorted(uniq)


def split_dataset_uniform(dataset: List[Dict[str, Any]], test_ratio: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Uniform split dataset into (train, test) and attach original line number as `index`.

    - test_ratio in (0,1)
    - The `index` field is 1-based (first line -> index=1)
    """
    n = len(dataset)
    if n == 0:
        return [], []
    k = max(1, int(n * test_ratio))
    test_indices_sorted = sorted(uniform_sample_indices(n, k))
    test_indices = set(test_indices_sorted)

    # Build test set with original index (1-based)
    test_set: List[Dict[str, Any]] = []
    for i in test_indices_sorted:
        obj = dict(dataset[i])  # shallow copy to avoid mutating original
        obj["index"] = i + 1
        test_set.append(obj)

    # Build train set with original index (1-based)
    train_set: List[Dict[str, Any]] = []
    for i in range(n):
        if i in test_indices:
            continue
        obj = dict(dataset[i])
        obj["index"] = i + 1
        train_set.append(obj)

    return train_set, test_set


def main():
    runtime_dir = Path(__file__).resolve().parents[1]
    default_input = runtime_dir / "GSM8k/grade-school-math/grade_school_math/data/train.jsonl"
    default_out_dir = runtime_dir / "GSM8k/grade-school-math/grade_school_math/data"

    ap = argparse.ArgumentParser(description="Uniform subsample JSONL into train/test splits")
    ap.add_argument("--input", type=Path, default=default_input, help=f"Input JSONL (default: {default_input})")
    ap.add_argument("--out-dir", type=Path, default=default_out_dir, help=f"Output dir for splits (default: {default_out_dir})")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio in (0,1), default 0.2")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    data = load_jsonl(str(args.input))
    train_set , test_set = split_dataset_uniform(data, test_ratio=args.test_ratio)

    ensure_dir(str(args.out_dir))
    train_jsonl = args.out_dir / "train_0.8.jsonl"
    test_jsonl = args.out_dir / "train_0.2.jsonl"

    with train_jsonl.open("w", encoding="utf-8") as wf:
        for obj in train_set:
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with test_jsonl.open("w", encoding="utf-8") as wf:
        for obj in test_set:
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved splits to: {args.out_dir}\n - train: {len(train_set)} \n - test:  {len(test_set)}")


if __name__ == "__main__":
    main()
