"""sentence_transition_network.py
Sentence-level sentiment transition network.

Similar to paragraph_transition but input is results/sentence_flow.csv.
Outputs:
    results/network_sentence/sentiment_transition_edges.csv
    results/network_sentence/sentiment_transition_nodes.csv

CLI flag: Called from main.py when using --sentence-transition.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from typing import Optional

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    _HAS_NX = False


def build_sentence_transition_network(
    sentence_csv: str = 'results/sentence_flow.csv',
    out_dir: str = 'results/network_sentence',
    min_count: int = 1,
    use_norm_score: bool = False,
) -> Optional[Path]:
    if not _HAS_NX:
        print('[INFO] networkx not installed → skipping sentence transition network')
        return None
    if not os.path.exists(sentence_csv):
        print(f'[INFO] sentence csv not found: {sentence_csv}')
        return None
    df = pd.read_csv(sentence_csv)
    req = {'article_index','sentence_index','label'}
    if not req.issubset(df.columns):
        print('[WARN] required columns missing for sentence transition network')
        return None
    if df.empty:
        print('[INFO] sentence csv empty')
        return None
    df_sorted = df.sort_values(['article_index','sentence_index'])
    df_sorted['next_label'] = df_sorted.groupby('article_index')['label'].shift(-1)
    transitions = df_sorted.dropna(subset=['next_label']).copy()
    transitions['next_label'] = transitions['next_label'].astype(str)
    transitions['label'] = transitions['label'].astype(str)
    edge_counts = transitions.groupby(['label','next_label']).size().reset_index(name='count')
    edge_counts = edge_counts[edge_counts['count'] >= min_count]
    if edge_counts.empty:
        print('[INFO] No sentence transitions meeting min_count')
        return None
    edge_counts['row_sum'] = edge_counts.groupby('label')['count'].transform('sum')
    edge_counts['prob'] = edge_counts['count'] / edge_counts['row_sum']
    G = nx.DiGraph()
    for _, r in edge_counts.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r['prob'])
    pagerank = nx.pagerank(G, weight='weight') if G.number_of_edges() > 0 else {}
    avg_norm_map = None
    if use_norm_score and 'score_norm' in df_sorted.columns:
        avg_norm_map = df_sorted.groupby('label')['score_norm'].mean().to_dict()
    nodes = []
    for n in G.nodes():
        nodes.append({
            'label': n,
            'in_degree': G.in_degree(n, weight='weight'),
            'out_degree': G.out_degree(n, weight='weight'),
            'total_degree': G.degree(n, weight='weight'),
            'pagerank': pagerank.get(n,0.0),
            'avg_norm_score': avg_norm_map.get(n) if avg_norm_map else None,
        })
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir_p / 'sentiment_transition_edges.csv'
    nodes_path = out_dir_p / 'sentiment_transition_nodes.csv'
    edge_counts[['label','next_label','count','prob']].to_csv(edges_path, index=False)
    pd.DataFrame(nodes).to_csv(nodes_path, index=False)
    print(f'[INFO] Sentence transition network edges -> {edges_path}')
    print(f'[INFO] Sentence transition network nodes -> {nodes_path}')
    return edges_path


if __name__ == '__main__':
    build_sentence_transition_network()
