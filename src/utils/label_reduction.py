"""label_reduction.py
Utility helpers to collapse 5 sentiment labels into 3 (positive, negative, sympathy) inline in pipeline.
If probability columns prob_<label> exist, they are aggregated to the new reduced labels
(by summing constituent original probs) and re-normalized.

Functions:
    load_mapping(path) -> dict
    reduce_sentiment_file(path, map_path) -> str (returns path)
    reduce_flow_file(path, map_path) -> Optional[str]

Side effects: overwrites the CSV in-place adding/overwriting column 'label'. Backs up original
as <name>_orig_labels.csv for traceability.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
from typing import Dict, List

REDUCED_COL = 'label'
BACKUP_SUFFIX = '_orig_labels'


def load_mapping(map_path: str | Path) -> Dict[str, str]:
    with open(map_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _backup(path: Path):
    b = path.with_name(path.stem + BACKUP_SUFFIX + path.suffix)
    if not b.exists():
        try:
            os.replace(path, b)
            # write back copy to original name
            df = pd.read_csv(b)
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"[WARN] Backup failed for {path}: {e}")


def _aggregate_probs(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    # Find prob_ columns
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if not prob_cols:
        return df
    # Reconstruct probability mass per reduced label
    # Extract original label names from column names: prob_<orig>
    from collections import defaultdict
    reduced_sums = defaultdict(list)
    for c in prob_cols:
        orig = c[len('prob_'):]
        red = mapping.get(orig)
        if red:
            reduced_sums[red].append(c)
    if not reduced_sums:
        return df
    # Create new aggregated columns prob_<reduced>
    for red, cols in reduced_sums.items():
        df[f'prob_{red}'] = df[cols].sum(axis=1)
    # Optional: normalize so that only reduced prob columns sum to 1 (if all present)
    reduced_prob_cols = [f'prob_{r}' for r in reduced_sums.keys()]
    total = df[reduced_prob_cols].sum(axis=1)
    # Avoid divide-by-zero
    total = total.replace({0: 1})
    for c in reduced_prob_cols:
        df[c] = df[c] / total
    return df


def _apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if 'label' not in df.columns:
        print('[WARN] No label column to reduce.')
        return df
    df['label'] = df['label'].map(lambda x: mapping.get(str(x), x))
    return df


def reduce_generic_file(csv_path: str | Path, map_path: str | Path) -> str:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    mapping = load_mapping(map_path)
    df = pd.read_csv(path)
    if df.empty:
        return str(path)
    # backup once
    _backup(path)
    df = _apply_mapping(df, mapping)
    df = _aggregate_probs(df, mapping)
    df.to_csv(path, index=False)
    print(f"[INFO] Reduced labels applied -> {path}")
    return str(path)


def reduce_sentiment_file(sentiment_csv: str | Path, map_path: str | Path) -> str:
    return reduce_generic_file(sentiment_csv, map_path)


def reduce_flow_file(flow_csv: str | Path, map_path: str | Path) -> str:
    return reduce_generic_file(flow_csv, map_path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--map', default='config/label_map_3.json')
    args = ap.parse_args()
    reduce_generic_file(args.file, args.map)
