"""paragraph_flow.py
Generate paragraph-level sentiment curve.
- Split article content into paragraphs
- Calculate paragraph scores using the selected sentiment analysis method (lexicon / transformer / zero-shot)
- Normalize and save curve data

Pipeline:
1. Input: processed articles (articles_processed.csv)
2. Paragraph splitting: based on empty lines, or fixed-length slicing if not available
3. Sentiment score: lightweight per-paragraph call instead of reusing sentiment_model.analyze_sentiment
4. Output: results/paragraph_flow.csv
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional
import math
import pandas as pd

from .sentiment_model import (
    load_lexicon,
    score_text_lexicon,
    TransformerSentiment,
    zero_shot_classify,
)


def _split_paragraphs(text: str, min_chars: int = 180) -> List[str]:
    if not text:
        return []
    # First, split by empty lines
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    if len(parts) <= 1:  # If there are almost no empty lines, use fixed-length slicing
        chunked = []
        buf = []
        count = 0
        for token in text.split():
            buf.append(token)
            count += len(token) + 1
            if count >= min_chars:
                chunked.append(' '.join(buf).strip())
                buf = []
                count = 0
        if buf:
            chunked.append(' '.join(buf).strip())
        parts = chunked
    return [p for p in parts if p]


def build_paragraph_flow(
    processed_dir: str = 'data/processed',
    out_csv: str = 'results/paragraph_flow.csv',
    method: str = 'lexicon',
    lexicon_path: str = 'models/sentiment_lexicon_ko.tsv',
    transformer_model: str = 'brainbert/kcbert-base-sentiment',
    zero_shot_labels: Optional[str] = None,
    zero_shot_model: str = 'joeddav/xlm-roberta-large-xnli',
    limit_articles: Optional[int] = None,
) -> Path:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    proc_file = Path(processed_dir) / 'articles_processed.csv'
    if not proc_file.exists():
        raise FileNotFoundError(proc_file)

    df = pd.read_csv(proc_file)
    if limit_articles:
        df = df.head(limit_articles)

    # Prepare analyzers
    lexicon = None
    transformer = None
    zero_labels_list: Optional[List[str]] = None
    if method == 'lexicon':
        lexicon = load_lexicon(lexicon_path)
    elif method in ('transformer', 'transformer-multi'):
        transformer = TransformerSentiment(model_name=transformer_model)
    elif method == 'zero-shot':
        if not zero_shot_labels:
            raise ValueError('zero_shot_labels required for zero-shot method')
        zero_labels_list = [l.strip() for l in zero_shot_labels.split(',') if l.strip()]
    else:
        raise ValueError('Unsupported method for paragraph flow')

    rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        article_id = idx
        content = row.get('clean_content') or row.get('content') or ''
        paragraphs = _split_paragraphs(str(content))
        if not paragraphs:
            continue

        # zero-shot batch once
        zs_results = None
        if method == 'zero-shot':
            zs_results = zero_shot_classify(paragraphs, zero_labels_list, model_name=zero_shot_model, batch_size=4)

        for p_idx, paragraph in enumerate(paragraphs):
            if method == 'lexicon':
                score = score_text_lexicon(paragraph, lexicon) if lexicon else 0.0
                label = 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
            elif method in ('transformer', 'transformer-multi'):
                pred = transformer.predict(paragraph[:1000]) if transformer else {"score": 0.0}
                score = float(pred.get('score', 0.0))
                label = pred.get('top_label') or ('positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral'))
            elif method == 'zero-shot':
                zr = zs_results[p_idx]
                score = float(zr.get('score', 0.0))
                label = zr.get('top_label')
            else:
                score = 0.0
                label = 'neutral'

            rows.append({
                'article_index': article_id,
                'paragraph_index': p_idx,
                'paragraph_text': paragraph,
                'score': score,
                'label': label,
                'source': row.get('source'),
                'date': row.get('date'),
                'title': row.get('title'),
            })

    out_df = pd.DataFrame(rows)
    # Normalization: z-score per article
    if not out_df.empty:
        out_df['score_norm'] = out_df.groupby('article_index')['score'].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-6)
        )
    out_path = Path(out_csv)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Paragraph flow saved -> {out_path}")
    return out_path


if __name__ == '__main__':
    build_paragraph_flow(method='lexicon')
