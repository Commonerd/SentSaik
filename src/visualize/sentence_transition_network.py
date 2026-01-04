"""Minimal module: sentence-level sentiment transition network visualization.

Usage:
    from src.visualize.sentence_transition_network import plot_sentence_transition_network
    plot_sentence_transition_network()

Outputs:
    results/plots_network_sentence/sentence_transition_network.png
    results/plots_network_sentence/sentence_transition_matrix.png
    (if INTERACTIVE=1) results/plots_network_sentence/interactive/sentence_transition_network_interactive.html
"""
from __future__ import annotations
import os
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False

try:
    import plotly.graph_objects as go  # type: ignore
    _PLOTLY = True
except Exception:  # pragma: no cover
    _PLOTLY = False


def _ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def plot_sentence_transition_network(
    edges_csv: str = 'results/network_sentence/sentiment_transition_edges.csv',
    nodes_csv: str = 'results/network_sentence/sentiment_transition_nodes.csv',
    out_dir: str = 'results/plots_network_sentence',
    highlight_top_k: int = 0,
):
    """Generate sentence-level sentiment transition network (static PNG + matrix + optional interactive HTML).

    Parameters
    ----------
    edges_csv : str
        ha='left', va='top', transform=ax.transAxes, fontsize=9, color='#222',
    nodes_csv : str
        CSV path with node-level metrics (optional columns: pagerank).
    fig.tight_layout(rect=[0, 0.07, 1, 1])
        Output directory for saved plots.
    highlight_top_k : int
        If >0, highlight top-k outgoing edges per source label.
    """
    if not _HAS_NX:
        print('[WARN] networkx unavailable; skipping sentence transition network.')
        return
    if not (os.path.exists(edges_csv) and os.path.exists(nodes_csv)):
        print(f'[WARN] Missing edges or nodes csv: {edges_csv} | {nodes_csv}')
        return
    out_p = _ensure_dir(out_dir)
    edges = pd.read_csv(edges_csv)
    nodes = pd.read_csv(nodes_csv)
    if edges.empty or nodes.empty:
        print('[INFO] Empty edges or nodes; nothing to plot.')
        return

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r.get('prob', 0.0))

    semantic_pos = {
        '부정': (-1.0, 0.0),
        '동정': (0.0, 1.2),
        '긍정': (1.0, 0.0),
    }
    spring_pos = nx.spring_layout(G, weight='weight', seed=24)
    pos = {n: semantic_pos.get(n, spring_pos.get(n, (0.0, 0.0))) for n in G.nodes()}
    # Geometric center for deciding inner(closer to center) vs outer(away)
    center_x = float(np.mean([xy[0] for xy in pos.values()])) if pos else 0.0
    center_y = float(np.mean([xy[1] for xy in pos.values()])) if pos else 0.0
    # Explicit curvature sign mapping per directed pair (both Korean and English labels)
    sign_map = {
        # Swap signs so each direction in a pair is on the opposite side
        ('부정','동정'): -1, ('동정','부정'): -1, ('긍정','동정'): 1, ('동정','긍정'): 1,
        ('Negative','Sympathy'): -1, ('Sympathy','Negative'): -1, ('Positive','Sympathy'): 1, ('Sympathy','Positive'): 1,
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    width_scaled = [1.0 + 4.0 * (w / max_w) for w in weights]
    probs = [G[u][v].get('prob', 0.0) for u, v in G.edges()]
    # Custom colormap so that the lowest edge probability color matches actual low edge tone rather than very pale
    min_p = min(probs) if probs else 0.0
    max_p = max(probs) if probs else 1.0
    base_cmap = plt.cm.Greens
    def prob_color(p: float):
        if max_p - min_p < 1e-9:
            return base_cmap(0.55)
        # Normalize p within observed range, then compress to avoid ultra-light colors
        norm = (p - min_p) / (max_p - min_p)
        adj = 0.25 + 0.75 * norm  # push away from 0 to keep visible
        return base_cmap(adj)
    base_cols = [prob_color(p) for p in probs]
    # Increase minimum visibility: raise base alpha & slightly darken very low probability edges
    edge_colors = []
    for (r, g, b, _), p in zip(base_cols, probs):
        adj_p = min(1.0, p)
        alpha = 0.38 + 0.62 * adj_p  # was 0.25 + 0.7*p
        # Darken low probabilities by slight desaturation shift toward mid-tone green
        if adj_p < 0.15:
            mix = 0.35  # mix with a darker greenish tone
            r2, g2, b2 = 0.1, 0.35, 0.1
            r = r * (1-mix) + r2 * mix
            g = g * (1-mix) + g2 * mix
            b = b * (1-mix) + b2 * mix
        edge_colors.append((r, g, b, alpha))

    highlight_edges: set[tuple[str, str]] = set()
    if highlight_top_k > 0:
        for src in edges['label'].unique():
            sub_top = edges[edges['label'] == src].sort_values('count', ascending=False).head(highlight_top_k)
            for _, row in sub_top.iterrows():
                highlight_edges.add((row['label'], row['next_label']))

    draw_colors = []
    for (u, v), base_col in zip(G.edges(), edge_colors):
        if highlight_edges:
            draw_colors.append(base_col if (u, v) in highlight_edges else (*base_col[:3], 0.25))
        else:
            draw_colors.append(base_col)

    undirected_pairs: dict[tuple[str, str], list[tuple[str, str, float, tuple[float, float, float, float]]]] = {}
    for (u, v), w, col in zip(G.edges(), width_scaled, draw_colors):
        key = tuple(sorted([u, v])) if u != v else (u, v)
        undirected_pairs.setdefault(key, []).append((u, v, w, col))

    base_rad = 0.22
    # Factors to differentiate inner/outer arcs for bidirectional pairs
    outer_factor = 0.55  # larger curvature (outer arc)
    inner_factor = 0.32  # smaller curvature (inner arc)
    for key, edge_list in undirected_pairs.items():
        # If bidirectional, assign deterministic ordering (by combined key then direction) for stable styling
        if len(edge_list) == 2:
            # Sort so heavier weight gets inner (shorter) curvature for readability (optional design choice)
            edge_list = sorted(edge_list, key=lambda t: (-t[2], t[0], t[1]))
            # Annotate with arc role
            annotated = []
            for idx, tpl in enumerate(edge_list):
                role = 'inner' if idx == 0 else 'outer'
                annotated.append((*tpl, role))
            edge_list_proc = annotated
        else:
            edge_list_proc = [(*tpl, 'single') for tpl in edge_list]
        for (u, v, w, col, role) in edge_list_proc:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            if u == v:
                # Self-loop: adjust orientation per node; for '동정' shift to right side to stay inside box
                loop_radius = 0.18
                if u == '동정':
                    theta = np.linspace(-0.15 * math.pi, 0.50 * math.pi, 42)  # slightly tighter right-upper arc
                    center_shift_x, center_shift_y = 0.10, 0.0
                elif u == '긍정':  # clockwise right-oriented loop for Positive
                    # Reverse the angle sweep so that the final point (arrowhead) is at the rightmost position
                    # Start near the upper-left (≈171°) and move clockwise to just below the positive x-axis (-9°)
                    theta = np.linspace(0.95 * math.pi, -0.05 * math.pi, 50)
                    center_shift_x, center_shift_y = 0.09, 0.0
                else:
                    theta = np.linspace(0.15 * math.pi, 0.85 * math.pi, 40)  # upper arc (wider)
                    center_shift_x, center_shift_y = 0.0, 0.0
                cx = x0 + center_shift_x + loop_radius * np.cos(theta)
                cy = y0 + center_shift_y + loop_radius * np.sin(theta)
                ax.plot(cx, cy, color=col, linewidth=max(1.2, w * 0.85), alpha=0.9)
                # Arrowhead at start->end direction (last segment tangential)
                end_x = cx[-1]; end_y = cy[-1]
                # Tangent for arrowhead
                if len(cx) >= 2:
                    tx = cx[-1] - cx[-2]; ty = cy[-1] - cy[-2]
                    mag_t = math.hypot(tx, ty) or 1.0
                    tx /= mag_t; ty /= mag_t
                else:
                    tx, ty = 0.0, 1.0
                ah_size = 0.042
                left_x = end_x - tx * ah_size + (-ty) * ah_size * 0.55
                left_y = end_y - ty * ah_size + (tx) * ah_size * 0.55
                right_x = end_x - tx * ah_size + (ty) * ah_size * 0.55
                right_y = end_y - ty * ah_size + (-tx) * ah_size * 0.55
                ax.fill([end_x, left_x, right_x], [end_y, left_y, right_y], color=col, alpha=0.95, linewidth=0)
                continue
            bidirectional = (role in {'inner','outer'})
            dx = x1 - x0; dy = y1 - y0
            length = math.hypot(dx, dy) or 1.0
            perp_x, perp_y = -dy / length, dx / length
            # Curvature magnitude selection
            if bidirectional:
                factor = outer_factor if role == 'outer' else inner_factor
            else:
                factor = 0.38  # single edge baseline
            curve_scale = factor * length * base_rad
            # Prefer explicit mapping; fall back to geometry/lexical
            if (u, v) in sign_map:
                sign = sign_map[(u, v)]
            elif bidirectional:
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                ctrl_plus_x = mid_x + perp_x * curve_scale
                ctrl_plus_y = mid_y + perp_y * curve_scale
                ctrl_minus_x = mid_x - perp_x * curve_scale
                ctrl_minus_y = mid_y - perp_y * curve_scale
                d_plus = math.hypot(ctrl_plus_x - center_x, ctrl_plus_y - center_y)
                d_minus = math.hypot(ctrl_minus_x - center_x, ctrl_minus_y - center_y)
                if role == 'inner':
                    sign = -1 if d_minus < d_plus else 1
                else:  # outer
                    sign = -1 if d_minus > d_plus else 1
            else:
                sign = 1 if (u < v) else -1
            control_x = (x0 + x1) / 2 + perp_x * curve_scale * sign
            control_y = (y0 + y1) / 2 + perp_y * curve_scale * sign
            t_vals = np.linspace(0, 1, 60)
            bx = (1 - t_vals) ** 2 * x0 + 2 * (1 - t_vals) * t_vals * control_x + t_vals ** 2 * x1
            by = (1 - t_vals) ** 2 * y0 + 2 * (1 - t_vals) * t_vals * control_y + t_vals ** 2 * y1
            ax.plot(bx, by, color=col, linewidth=w, alpha=0.92, solid_capstyle='round')
            # Arrowhead position: slightly different along t to reduce overlap (toward target end)
            t_mid = 0.62 if role == 'outer' else (0.58 if role == 'inner' else 0.60)
            mx = (1 - t_mid) ** 2 * x0 + 2 * (1 - t_mid) * t_mid * control_x + t_mid ** 2 * x1
            my = (1 - t_mid) ** 2 * y0 + 2 * (1 - t_mid) * t_mid * control_y + t_mid ** 2 * y1
            dx_dt = 2 * (1 - t_mid) * (control_x - x0) + 2 * t_mid * (x1 - control_x)
            dy_dt = 2 * (1 - t_mid) * (control_y - y0) + 2 * t_mid * (y1 - control_y)
            mag = math.hypot(dx_dt, dy_dt) or 1.0
            tx, ty = dx_dt / mag, dy_dt / mag
            ah_len = 0.058 * (1 + 0.4 * (w / max(width_scaled)))
            ah_w = ah_len * 0.52
            left_x = mx - tx * ah_len + (-ty) * ah_w
            left_y = my - ty * ah_len + (tx) * ah_w
            right_x = mx - tx * ah_len + (ty) * ah_w
            right_y = my - ty * ah_len + (-tx) * ah_w
            ax.fill([mx, left_x, right_x], [my, left_y, right_y], color=col, alpha=0.95, linewidth=0)

    sizes = [380 + 2400 * nodes.set_index('label').loc[n].get('pagerank', 0.0) if 'pagerank' in nodes.columns else 400 + 30 * G.degree(n, weight='weight') for n in G.nodes()]
    _label_color_map = {
        '긍정': (31 / 255, 119 / 255, 180 / 255, 0.78),
        '부정': (214 / 255, 39 / 255, 40 / 255, 0.78),
        '동정': (255 / 255, 219 / 255, 88 / 255, 0.85),
    }
    node_colors = [_label_color_map.get(n, (0.7, 0.7, 0.7, 0.6)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='#333', linewidths=0.9, node_size=sizes, ax=ax)
    en_map = {'긍정': 'Positive', '부정': 'Negative', '동정': 'Sympathy'}
    # Node labels (Korean→English)
    node_labels = {n: en_map.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, ax=ax, font_family='sans-serif')
    # Display in-degree (number of incoming edges) below each node
    for n in G.nodes():
        x, y = pos[n]
        in_deg = G.in_degree(n, weight='weight')
        # Sympathy node gets a larger vertical offset
        if n == '동정' or n == 'Sympathy':
            y_offset = 0.18
        else:
            y_offset = 0.13
        ax.text(x, y-y_offset, f"{in_deg}", fontsize=10, color='#222', ha='center', va='top', alpha=1.0, fontweight='bold')

    for key, edge_list in undirected_pairs.items():
        if not edge_list:
            continue
        # Force curve direction: Negative→Sympathy, Positive→Sympathy are inner; Sympathy→Negative, Sympathy→Positive are outer
        if len(edge_list) == 2:
            forced_roles = None
            if set(key) == {'부정', '동정'}:
                # Negative→Sympathy(inner), Sympathy→Negative(outer)
                forced_roles = {('부정','동정'): 'inner', ('동정','부정'): 'outer'}
            elif set(key) == {'긍정', '동정'}:
                # Positive→Sympathy(inner), Sympathy→Positive(outer)
                forced_roles = {('긍정','동정'): 'inner', ('동정','긍정'): 'outer'}
            if forced_roles:
                edge_list_proc = []
                for tpl in edge_list:
                    role = forced_roles.get((tpl[0], tpl[1]), 'outer')
                    edge_list_proc.append((*tpl, role))
            else:
                edge_list_tmp = sorted(edge_list, key=lambda t: (-t[2], t[0], t[1]))
                edge_list_proc = []
                for idx, tpl in enumerate(edge_list_tmp):
                    role = 'inner' if idx == 0 else 'outer'
                    edge_list_proc.append((*tpl, role))
        else:
            edge_list_proc = [(*tpl, 'single') for tpl in edge_list]
        for (u, v, w, col, role) in edge_list_proc:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            data = G[u][v]
            if u == v:
                if u == '동정':
                    ax.text(x0 + -0.165, y0 + 0.20, f"loop {data.get('weight', 0)}/{data.get('prob', 0):.2f}", fontsize=9,
                            ha='center', va='center', color='black', alpha=0.95,
                            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#666', alpha=0.55))
                else:
                    ax.text(x0, y0 + 0.22, f"loop {data.get('weight', 0)}/{data.get('prob', 0):.2f}", fontsize=9,
                            ha='center', va='center', color='black', alpha=0.95,
                            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#666', alpha=0.55))
                continue
            dx = x1 - x0; dy = y1 - y0
            length = math.hypot(dx, dy) or 1.0
            perp_x, perp_y = -dy / length, dx / length
            # Use the same geometry-based curvature sign as the first loop
            if role == 'outer':
                factor = outer_factor
                t_mid = 0.46
            elif role == 'inner':
                factor = inner_factor
                t_mid = 0.41
            else:
                factor = 0.38
                t_mid = 0.5
            curve_scale = factor * length * base_rad
            if (u, v) in sign_map:
                sign = sign_map[(u, v)]
            elif role in {'inner','outer'}:
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                ctrl_plus_x = mid_x + perp_x * curve_scale
                ctrl_plus_y = mid_y + perp_y * curve_scale
                ctrl_minus_x = mid_x - perp_x * curve_scale
                ctrl_minus_y = mid_y - perp_y * curve_scale
                d_plus = math.hypot(ctrl_plus_x - center_x, ctrl_plus_y - center_y)
                d_minus = math.hypot(ctrl_minus_x - center_x, ctrl_minus_y - center_y)
                if role == 'inner':
                    sign = -1 if d_minus < d_plus else 1
                else:  # outer
                    sign = -1 if d_minus > d_plus else 1
            else:
                sign = 1 if (u < v) else -1
            control_x = (x0 + x1) / 2 + perp_x * curve_scale * sign
            control_y = (y0 + y1) / 2 + perp_y * curve_scale * sign
            t_label = min(0.88, t_mid + 0.22)
            lx = (1 - t_label) ** 2 * x0 + 2 * (1 - t_label) * t_label * control_x + t_label ** 2 * x1
            ly = (1 - t_label) ** 2 * y0 + 2 * (1 - t_label) * t_label * control_y + t_label ** 2 * y1
            base_off = 0.045
            label_off = base_off * (1.1 if role == 'outer' else (0.85 if role == 'inner' else 1.0)) * sign
            label_x = lx + perp_x * label_off
            label_y = ly + perp_y * label_off
            ax.text(label_x, label_y, f"{data.get('weight', 0)}/{data.get('prob', 0):.2f}", fontsize=9,
                    ha='center', va='center', color='darkslategray', alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#666', alpha=0.55))

    sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=plt.Normalize(vmin=min_p, vmax=max_p))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Transition Probability')
    ax.set_title('Sentence-level Sentiment Transition: Hong Sa-ik in Korean Media(1947-2020)', pad=12)
    # Legend: node→edge→loop order, for readability
    # Legend: standard legend style, square box, left-aligned
    legend_text = (
        '• Node: total incoming count\n'
        '• Edge: transition count / transition probability\n'
        '• Loop: self-transition count / self-transition probability'
    )
    # Trick: use alignment baseline at center, but offset text left inside the box
    ax.text(0.5, -0.11, legend_text,
        ha='center', va='top', transform=ax.transAxes, fontsize=9, color='#222',
        linespacing=1.35, multialignment='left',
        bbox=dict(boxstyle='round,pad=0.32', fc='white', ec='#888', alpha=0.92))
    # Draw invisible left-aligned text for bbox sizing, then overlay visible text with ha='center' for box centering
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_p / 'sentence_transition_network.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Matrix
    mat = edges.pivot_table(index='label', columns='next_label', values='prob', fill_value=0)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='Greens')
    ax_curr = plt.gca()
    ax_curr.set_yticklabels([en_map.get(t.get_text(), t.get_text()) for t in ax_curr.get_yticklabels()], rotation=0)
    # X축 라벨을 수평(0도)으로
    ax_curr.set_xticklabels([en_map.get(t.get_text(), t.get_text()) for t in ax_curr.get_xticklabels()], rotation=0, ha='center')
    plt.title('Sentence Transition Probability Matrix')
    plt.xlabel('Next label')
    plt.ylabel('Current label')
    plt.text(0.5, -0.15, 'Row=current label, Column=next label', transform=ax_curr.transAxes,
             ha='center', va='top', fontsize=9, color='#444')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_p / 'sentence_transition_matrix.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Sentence transition network plots -> {out_p}')

    # Interactive
    if _PLOTLY and os.environ.get('INTERACTIVE', '0') in {'1', 'true', 'True'}:
        try:
            inter_dir = out_p / 'interactive'
            inter_dir.mkdir(exist_ok=True)
            label_en = {'긍정': 'Positive', '부정': 'Negative', '동정': 'Sympathy'}
            node_positions = {
                'Negative': (-1, 0),
                'Sympathy': (0, 1.2),
                'Positive': (1, 0)
            }
            node_list = [n for n in G.nodes() if label_en.get(n, n) in node_positions]
            x_vals = [node_positions[label_en.get(n, n)][0] for n in node_list]
            y_vals = [node_positions[label_en.get(n, n)][1] for n in node_list]
            color_map_hex = {
                'Positive': 'rgba(31,119,180,0.78)',
                'Negative': 'rgba(214,39,40,0.78)',
                'Sympathy': 'rgba(255,219,88,0.85)'
            }
            node_labels_en = [label_en.get(n, n) for n in node_list]
            node_colors_hex = [color_map_hex.get(label_en.get(n, n), 'rgba(180,180,180,0.6)') for n in node_list]
            size_map = {n: s for n, s in zip(G.nodes(), sizes)}
            node_sizes = [size_map[n] * 0.06 for n in node_list]
            edge_traces = []
            label_annotations = []
            total_edges = len(G.edges())
            base_offset = 0.035 + 0.018 * (total_edges / 10) ** 0.5
            for (u, v) in G.edges():
                u_en = label_en.get(u, u)
                v_en = label_en.get(v, v)
                if u_en not in node_positions or v_en not in node_positions:
                    continue
                x0, y0 = node_positions[u_en]
                x1, y1 = node_positions[v_en]
                data = G[u][v]
                if u == v:
                    loop_off = 0.18
                    label_annotations.append(dict(
                        x=x0, y=y0 + loop_off, text=f"loop {data.get('weight', 0)}/{data.get('prob', 0):.2f}",
                        showarrow=False, font=dict(size=10, color='black'),
                        bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(80,80,80,0.4)', borderwidth=0.5
                    ))
                    continue
                dx = x1 - x0; dy = y1 - y0
                length = math.hypot(dx, dy) or 1.0
                ux, uy = dx / length, dy / length
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                p = float(data.get('prob', 0.0))
                alpha_line = 0.25 + 0.7 * max(0.0, min(1.0, p))
                col_line = f'rgba(100,100,100,{alpha_line:.3f})'
                edge_traces.append(go.Scatter(x=[x0, mid_x], y=[y0, mid_y], mode='lines',
                                              line=dict(color=col_line, width=1.7),
                                              hoverinfo='skip', showlegend=False))
                ah_size = 0.07
                left_x = mid_x + (-uy) * ah_size
                left_y = mid_y + (ux) * ah_size
                right_x = mid_x + (uy) * ah_size
                right_y = mid_y + (-ux) * ah_size
                edge_traces.append(go.Scatter(x=[mid_x, left_x, right_x, mid_x], y=[mid_y, left_y, right_y, mid_y],
                                              mode='lines', fill='toself', line=dict(color=col_line, width=1),
                                              hoverinfo='skip', showlegend=False))
                perp_x, perp_y = -uy, ux
                has_reverse = G.has_edge(v, u)
                off = base_offset if (u < v or not has_reverse) else -base_offset
                if has_reverse and (v < u):
                    off = -off
                label_x = mid_x + perp_x * off
                label_y = mid_y + perp_y * off
                label_annotations.append(dict(
                    x=label_x, y=label_y,
                    text=f"{data.get('weight', 0)}/{data.get('prob', 0):.2f}",
                    showarrow=False,
                    font=dict(size=10, color='darkslategray'),
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(120,120,120,0.3)', borderwidth=0.5
                ))
            node_trace = go.Scatter(
                x=x_vals, y=y_vals, mode='markers+text',
                text=node_labels_en, textposition='middle center',
                marker=dict(size=node_sizes, color=node_colors_hex, line=dict(color='rgba(50,50,50,0.6)', width=1)),
                hoverinfo='text', showlegend=False
            )
            layout = go.Layout(
                title='Sentence  Level Sentiment Transition  (Interactive)',
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                annotations=label_annotations + [
                    dict(x=1.02, y=0.02, xref='paper', yref='paper', showarrow=False, align='left',
                         text='edge: count/prob<br>loop: self-transition', font=dict(size=9),
                         bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(120,120,120,0.4)', borderwidth=0.5)
                ],
                margin=dict(l=10, r=140, t=60, b=10),
                paper_bgcolor='white', plot_bgcolor='white'
            )
            fig_i = go.Figure(data=edge_traces + [node_trace], layout=layout)
            fig_i.write_html(inter_dir / 'sentence_transition_network_interactive.html')
        except Exception as e:  # pragma: no cover
            print('[WARN] Interactive sentence network failed:', e)


def plot_contextual_sentence_transition_network(
    edges_csv: str = 'results/network_context_sentence/sentiment_transition_edges.csv',
    nodes_csv: str = 'results/network_context_sentence/sentiment_transition_nodes.csv',
    out_dir: str = 'results/plots_network_context_sentence',
    highlight_top_k: int = 0,
):
    """Generate context-based sentence-level sentiment transition network (static PNG + matrix + optional interactive HTML)."""
    return plot_sentence_transition_network(
        edges_csv=edges_csv,
        nodes_csv=nodes_csv,
        out_dir=out_dir,
        highlight_top_k=highlight_top_k,
    )

__all__ = ['plot_sentence_transition_network', 'plot_contextual_sentence_transition_network']
