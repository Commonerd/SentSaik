"""sentiment_transition_network.py
Generate Sentiment Transition Network.

Goals:
- Count (label_t -> label_{t+1}) transitions within the same article from paragraph flow (`paragraph_flow.csv`)
- Directed graph based on transition weights (NetworkX)
- Node metrics: in_degree, out_degree, weighted_degree, PageRank
- Edge metrics: count, normalized_probability (row-normalized)

Outputs:
- results/network_paragraph/sentiment_transition_edges.csv
- results/network_paragraph/sentiment_transition_nodes.csv

Additional usage:
- Network plot / chord-like layout in visualization models
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


def build_sentiment_transition_network(
    paragraph_csv: str = 'results/paragraph_flow.csv',
    out_dir: str = 'results/network_paragraph',
    min_count: int = 1,
    use_norm_score: bool = False,
) -> Optional[Path]:
    """Build sentiment transition network from paragraph_flow.

    Parameters
    ----------
    paragraph_csv : str
        Path to paragraph_flow.csv (requires columns: article_index, paragraph_index, label).
    out_dir : str
        Output directory for edge/node CSVs.
    min_count : int
        Minimum transition count to retain an edge.
    use_norm_score : bool
        If True, optionally compute avg normalized score per node.
    """
    if not _HAS_NX:
        print('[INFO] networkx not installed → skipping transition network')
        return None
    if not os.path.exists(paragraph_csv):
        print(f'[INFO] paragraph csv not found: {paragraph_csv}')
        return None
    df = pd.read_csv(paragraph_csv)
    req = {'article_index','paragraph_index','label'}
    if not req.issubset(df.columns):
        print('[WARN] required columns missing for transition network')
        return None
    if df.empty:
        print('[INFO] paragraph csv empty')
        return None

    # Sort and extract transitions
    df_sorted = df.sort_values(['article_index','paragraph_index'])
    df_sorted['next_label'] = df_sorted.groupby('article_index')['label'].shift(-1)
    transitions = df_sorted.dropna(subset=['next_label']).copy()
    transitions['next_label'] = transitions['next_label'].astype(str)
    transitions['label'] = transitions['label'].astype(str)

    edge_counts = transitions.groupby(['label','next_label']).size().reset_index(name='count')
    edge_counts = edge_counts[edge_counts['count'] >= min_count]

    if edge_counts.empty:
        print('[INFO] No transitions meeting min_count')
        return None

    # Row (source label) based probability normalization
    edge_counts['row_sum'] = edge_counts.groupby('label')['count'].transform('sum')
    edge_counts['prob'] = edge_counts['count'] / edge_counts['row_sum']

    # Build graph
    G = nx.DiGraph()
    for _, r in edge_counts.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r['prob'])

    # Node statistics
    nodes = []
    pagerank = nx.pagerank(G, weight='weight') if G.number_of_edges() > 0 else {}
    # Additional network metrics: betweenness (weighted), in/out strength, clustering (converted to undirected)
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True) if G.number_of_edges() > 0 else {}
    except Exception:
        betweenness = {}
    try:
        undirected = G.to_undirected()
        clustering = nx.clustering(undirected, weight='weight') if undirected.number_of_edges() > 0 else {}
    except Exception:
        clustering = {}
    avg_norm_map = None
    if use_norm_score and 'score_norm' in df_sorted.columns:
        avg_norm_map = df_sorted.groupby('label')['score_norm'].mean().to_dict()
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
            'avg_norm_score': avg_norm_map.get(n) if avg_norm_map else None,
        })

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir_p / 'sentiment_transition_edges.csv'
    nodes_path = out_dir_p / 'sentiment_transition_nodes.csv'
    # 보기 편한 퍼센트 컬럼도 추가
    edge_counts['prob_pct'] = (edge_counts['prob'] * 100).round(2)
    edge_counts[['label','next_label','count','prob','prob_pct']].to_csv(edges_path, index=False)
    pd.DataFrame(nodes).to_csv(nodes_path, index=False)
    print(f'[INFO] Transition network edges -> {edges_path}')
    print(f'[INFO] Transition network nodes -> {nodes_path}')
    return edges_path


def _extract_month_key(df):
    if 'date' in df.columns:
        try:
            d = pd.to_datetime(df['date'], errors='coerce')
            return d.dt.to_period('M').astype(str)
        except Exception:
            return pd.Series(['unknown']*len(df))
    return pd.Series(['unknown']*len(df))


def build_monthly_transition_networks(
    paragraph_csv: str = 'results/paragraph_flow.csv',
    out_root: str = 'results/network_paragraph/monthly',
    min_count: int = 1,
    highlight_top_k: int = 0,
    max_months: int = 6,
):
    """Generate monthly sentiment transition edge/node CSVs (recent months first) and optional highlight info.

    highlight_top_k: if >0 produce for each month a filtered edges_highlight CSV with top-k outgoing transitions per source label.
    """
    if not _HAS_NX:
        print('[INFO] networkx 미설치: monthly transitions skip')
        return
    if not os.path.exists(paragraph_csv):
        return
    df = pd.read_csv(paragraph_csv)
    if df.empty or 'label' not in df.columns:
        return
    month_key = _extract_month_key(df)
    df['__month'] = month_key
    months = [m for m in month_key.dropna().unique() if m != 'unknown']
    if not months:
        return
    # 최근(month 문자열 정렬)에서 역순
    months_sorted = sorted(months)[-max_months:]
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)
    for m in months_sorted:
        sub = df[df['__month'] == m].copy()
        if sub.empty: 
            continue
        # reuse logic: build transitions in-memory
        sub = sub.sort_values(['article_index','paragraph_index'])
        sub['next_label'] = sub.groupby('article_index')['label'].shift(-1)
        transitions = sub.dropna(subset=['next_label']).copy()
        transitions['next_label'] = transitions['next_label'].astype(str)
        transitions['label'] = transitions['label'].astype(str)
        edge_counts = transitions.groupby(['label','next_label']).size().reset_index(name='count')
        edge_counts = edge_counts[edge_counts['count'] >= min_count]
        if edge_counts.empty:
            continue
        edge_counts['row_sum'] = edge_counts.groupby('label')['count'].transform('sum')
        edge_counts['prob'] = edge_counts['count'] / edge_counts['row_sum']
        G = nx.DiGraph()
        for _, r in edge_counts.iterrows():
            G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r['prob'])
        pagerank = nx.pagerank(G, weight='weight') if G.number_of_edges() else {}
        nodes = []
        for n in G.nodes():
            nodes.append({
                'label': n,
                'in_degree': G.in_degree(n, weight='weight'),
                'out_degree': G.out_degree(n, weight='weight'),
                'total_degree': G.degree(n, weight='weight'),
                'pagerank': pagerank.get(n,0.0)
            })
        m_dir = out_root_p / m
        m_dir.mkdir(parents=True, exist_ok=True)
        edge_counts[['label','next_label','count','prob']].to_csv(m_dir/'edges.csv', index=False)
        pd.DataFrame(nodes).to_csv(m_dir/'nodes.csv', index=False)
        if highlight_top_k > 0:
            hi_rows = []
            for lab, grp in edge_counts.groupby('label'):
                top = grp.sort_values('count', ascending=False).head(highlight_top_k)
                hi_rows.append(top)
            if hi_rows:
                pd.concat(hi_rows).to_csv(m_dir/'edges_highlight.csv', index=False)
        print(f'[INFO] Monthly transition network -> {m_dir}')


if __name__ == '__main__':
    build_sentiment_transition_network()
