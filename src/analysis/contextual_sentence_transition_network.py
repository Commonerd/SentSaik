"""
contextual_sentence_transition_network.py
Generate transition network (edge/node csv) from context-based sentence sentiment results (contextual_sentence_flow.csv)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from typing import Optional

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

def build_contextual_sentence_transition_network(
    flow_csv: str = 'results/contextual_sentence_flow.csv',
    out_dir: str = 'results/network_context_sentence',
    min_count: int = 1,
) -> Optional[Path]:
    if not _HAS_NX:
        print('[INFO] networkx not installed → skipping transition network')
        return None
    if not os.path.exists(flow_csv):
        print(f'[INFO] flow csv not found: {flow_csv}')
        return None
    df = pd.read_csv(flow_csv)
    req = {'article_index','sentence_index','label'}
    if not req.issubset(df.columns):
        print('[WARN] required columns missing for transition network')
        return None
    if df.empty:
        print('[INFO] flow csv empty')
        return None
    # Sort and extract transitions
    df_sorted = df.sort_values(['article_index','sentence_index'])
    df_sorted['next_label'] = df_sorted.groupby('article_index')['label'].shift(-1)
    transitions = df_sorted.dropna(subset=['next_label']).copy()
    transitions['next_label'] = transitions['next_label'].astype(str)
    transitions['label'] = transitions['label'].astype(str)
    edge_counts = transitions.groupby(['label','next_label']).size().reset_index(name='count')
    edge_counts = edge_counts[edge_counts['count'] >= min_count]
    if edge_counts.empty:
        print('[INFO] No transitions meeting min_count')
        return None
    edge_counts['row_sum'] = edge_counts.groupby('label')['count'].transform('sum')
    edge_counts['prob'] = edge_counts['count'] / edge_counts['row_sum']
    G = nx.DiGraph()
    for _, r in edge_counts.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r['prob'])
    nodes = []
    pagerank = nx.pagerank(G, weight='weight') if G.number_of_edges() > 0 else {}
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True) if G.number_of_edges() > 0 else {}
    except Exception:
        betweenness = {}
    try:
        undirected = G.to_undirected()
        clustering = nx.clustering(undirected, weight='weight') if undirected.number_of_edges() > 0 else {}
    except Exception:
        clustering = {}
    for n in G.nodes():
        in_w = G.in_degree(n, weight='weight')
        out_w = G.out_degree(n, weight='weight')
        nodes.append({
            'label': n,
            'in_degree': in_w,
            'out_degree': out_w,
            'total_degree': in_w + out_w,
            'pagerank': pagerank.get(n, 0.0),
            'betweenness': betweenness.get(n, 0.0),
            'clustering': clustering.get(n, 0.0),
        })
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir_p / 'sentiment_transition_edges.csv'
    nodes_path = out_dir_p / 'sentiment_transition_nodes.csv'
    edge_counts['prob_pct'] = (edge_counts['prob'] * 100).round(2)
    edge_counts[['label','next_label','count','prob','prob_pct']].to_csv(edges_path, index=False)
    pd.DataFrame(nodes).to_csv(nodes_path, index=False)
    print(f'[INFO] Contextual sentence transition edges -> {edges_path}')
    print(f'[INFO] Contextual sentence transition nodes -> {nodes_path}')
    return edges_path

if __name__ == '__main__':
    build_contextual_sentence_transition_network()
