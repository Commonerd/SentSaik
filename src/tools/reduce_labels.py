"""reduce_labels.py
Collapse 5 sentiment labels into 3 (positive, negative, sympathy) from original sentiment/flow files.
Mapping rules use config/label_map_3.json.

Outputs:
    results/sentiment_reduced.csv
    (if exists) results/paragraph_flow_reduced.csv
    (if exists) results/sentence_flow_reduced.csv
Optional: If --rebuild-transition is used, recalculate transition network (default: paragraph level only)

Usage examples:
    python -m src.tools.reduce_labels --map config/label_map_3.json --rebuild-transition
    python -m src.tools.reduce_labels --map config/label_map_3.json --plots

Options:
    --plots : Generate simple key plots (label count & transition network) based on reduced labels

Note: The original file is not modified; *_reduced.csv is newly created.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd


def load_map(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_mapping(df: pd.DataFrame, mapping: dict, new_col: str = 'label_reduced') -> pd.DataFrame:
    if 'label' not in df.columns:
        raise ValueError('label column not found')
    df[new_col] = df['label'].map(mapping)
    return df

def reduce_file(in_path: Path, out_path: Path, mapping: dict, new_col: str = 'label_reduced'):
    if not in_path.exists():
        return False
    df = pd.read_csv(in_path)
    if df.empty:
        return False
    if 'label' not in df.columns:
        return False
    df = apply_mapping(df, mapping, new_col=new_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Reduced labels -> {out_path}")
    return True

def rebuild_transition_from_paragraph(reduced_paragraph_csv: Path, out_dir: Path = Path('results/network_reduced'), min_count: int = 1):
    """Rebuild transition network (reduced labels)."""
    try:
        import networkx as nx  # type: ignore
    except Exception:
        print('[INFO] networkx not installed: transition rebuild skip')
        return
    if not reduced_paragraph_csv.exists():
        print('[INFO] paragraph_flow_reduced.csv not found: transition rebuild skip')
        return
    df = pd.read_csv(reduced_paragraph_csv)
    if {'label_reduced','article_index','paragraph_index'} - set(df.columns):
        print('[INFO] Required columns missing: transition rebuild skip')
        return
    df = df.sort_values(['article_index','paragraph_index'])
    # Adjacent transitions
    pairs = []
    for aid, g in df.groupby('article_index'):
        labs = g['label_reduced'].tolist()
        for a,b in zip(labs, labs[1:]):
            if a is None or b is None: continue
            pairs.append((a,b))
    if not pairs:
        print('[INFO] No transition pairs')
        return
    from collections import Counter
    ctr = Counter(pairs)
    # edge df
    import math
    edges_rows = []
    total_out = {}
    for (a,b), cnt in ctr.items():
        total_out.setdefault(a,0)
        total_out[a]+=cnt
    for (a,b), cnt in ctr.items():
        prob = cnt / total_out[a]
        edges_rows.append({'label': a, 'next_label': b, 'count': cnt, 'prob': prob})
    import pandas as _pd
    edges_df = _pd.DataFrame(edges_rows)
    # node stats (pagerank)
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'])
    try:
        pr = nx.pagerank(G, weight='weight')
    except Exception:
        pr = {n:0 for n in G.nodes()}
    nodes_df = _pd.DataFrame({'label': list(pr.keys()), 'pagerank': list(pr.values())})
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_dir/'sentiment_transition_edges.csv', index=False)
    nodes_df.to_csv(out_dir/'sentiment_transition_nodes.csv', index=False)
    print(f"[INFO] Reduced transition network -> {out_dir}")


def plot_reduced_counts(sentiment_reduced: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        print('[INFO] matplotlib/seaborn 미설치: plot skip')
        return
    if not sentiment_reduced.exists():
        return
    df = pd.read_csv(sentiment_reduced)
    if 'label_reduced' not in df.columns:
        return
    counts = df['label_reduced'].value_counts()
    plt.figure(figsize=(4.5,3.2))
    sns.barplot(x=counts.index, y=counts.values, palette='Set2')
    plt.title('Reduced Label Counts')
    plt.ylabel('Count')
    plt.tight_layout()
    out_path = sentiment_reduced.parent/'reduced_label_counts.png'
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[INFO] Reduced label count plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='config/label_map_3.json')
    ap.add_argument('--rebuild-transition', action='store_true')
    ap.add_argument('--min-count', type=int, default=1)
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()

    mapping = load_map(args.map)
    sent_ok = reduce_file(Path('results/sentiment.csv'), Path('results/sentiment_reduced.csv'), mapping)
    para_ok = reduce_file(Path('results/paragraph_flow.csv'), Path('results/paragraph_flow_reduced.csv'), mapping)
    sentc_ok = reduce_file(Path('results/sentence_flow.csv'), Path('results/sentence_flow_reduced.csv'), mapping)

    if args.rebuild_transition and para_ok:
        rebuild_transition_from_paragraph(Path('results/paragraph_flow_reduced.csv'), Path('results/network_reduced'), min_count=args.min_count)

    if args.plots and sent_ok:
        plot_reduced_counts(Path('results/sentiment_reduced.csv'))

    print('[DONE] Reduction complete.')

if __name__ == '__main__':
    main()
