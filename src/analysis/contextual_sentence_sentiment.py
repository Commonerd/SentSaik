"""
contextual_sentence_sentiment.py
Context-based sentence-level sentiment analysis
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
import pandas as pd

from .sentiment_model import (
    load_lexicon,
    score_text_lexicon,
    TransformerSentiment,
    zero_shot_classify,
)

 # Sentence splitting function (same as before)
def _split_sentences_regex(text: str, min_len: int = 4) -> List[str]:
    import re
    _SENT_SPLIT_RE = re.compile(r'(?<=[.!?。!?])\s+')
    if not text:
        return []
    raw_parts = _SENT_SPLIT_RE.split(text.strip())
    cleaned = [p.strip() for p in raw_parts if len(p.strip()) >= min_len]
    return cleaned


# Context-based sentiment prediction

def build_contextual_sentence_flow(
    processed_dir: str = 'data/processed',
    out_csv: str = 'results/contextual_sentence_flow.csv',
    method: str = 'lexicon',
    lexicon_path: str = 'models/sentiment_lexicon_ko.tsv',
    transformer_model: str = 'brainbert/kcbert-base-sentiment',
    zero_shot_labels: Optional[str] = None,
    zero_shot_model: str = 'joeddav/xlm-roberta-large-xnli',
    context_window: int = 1,  # How many previous/next sentences to include as context
) -> Path:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    proc_file = Path(processed_dir) / 'articles_processed.csv'
    if not proc_file.exists():
        raise FileNotFoundError(proc_file)

    df = pd.read_csv(proc_file)
    lexicon = None
    transformer = None
    zero_labels_list: Optional[List[str]] = None
    if method == 'lexicon':
        lexicon = load_lexicon(lexicon_path)
    elif method in ('transformer','transformer-multi'):
        transformer = TransformerSentiment(model_name=transformer_model)
    elif method == 'zero-shot':
        if not zero_shot_labels:
            raise ValueError('zero_shot_labels required for zero-shot method')
        zero_labels_list = [l.strip() for l in zero_shot_labels.split(',') if l.strip()]
    else:
        raise ValueError('Unsupported method for sentence flow')

    rows: List[Dict[str, object]] = []
    total = len(df)
    for idx, row in df.iterrows():
        article_id = idx
        content = row.get('clean_content') or row.get('content') or ''
        sentences = _split_sentences_regex(str(content), min_len=4)
        if not sentences:
            continue
        for s_idx, sent in enumerate(sentences):
            # Apply context window: include previous/next sentences
            left = max(0, s_idx - context_window)
            right = min(len(sentences), s_idx + context_window + 1)
            context = ' '.join(sentences[left:right])
            # Optionally emphasize the current sentence
            # context = context.replace(sent, f'[CTX]{sent}[/CTX]')
            if method == 'lexicon':
                score = score_text_lexicon(context, lexicon) if lexicon else 0.0
                label = 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
            elif method in ('transformer','transformer-multi'):
                pred = transformer.predict(context[:1000]) if transformer else {"score":0.0}
                score = float(pred.get('score',0.0))
                label = pred.get('top_label') or ('positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral'))
            elif method == 'zero-shot':
                zs_result = zero_shot_classify([context], zero_labels_list, model_name=zero_shot_model, batch_size=1)[0]
                score = float(zs_result.get('score',0.0))
                label = zs_result.get('top_label')
            else:
                score = 0.0
                label = 'neutral'
            print(f"[Contextual Sentiment] article:{article_id} sent_idx:{s_idx} label:{label} score:{score:.4f} text:{sent[:50]}", flush=True)
            rows.append({
                'article_index': article_id,
                'sentence_index': s_idx,
                'sentence_text': sent,
                'context_text': context,
                'score': score,
                'label': label,
                'source': row.get('source'),
                'date': row.get('date'),
                'title': row.get('title'),
            })
        # Print progress percentage
        if (idx + 1) % max(1, total // 100) == 0 or (idx + 1) == total:
            percent = int((idx + 1) / total * 100)
            print(f"[Progress] {idx + 1}/{total} ({percent}%) done", flush=True)
    out_df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Contextual sentence flow saved -> {out_path}")
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--out-csv', default='results/contextual_sentence_flow.csv')
    parser.add_argument('--method', choices=['lexicon','transformer','transformer-multi','zero-shot'], default='lexicon')
    parser.add_argument('--lexicon-path', default='models/sentiment_lexicon_ko.tsv')
    parser.add_argument('--transformer-model', default='brainbert/kcbert-base-sentiment')
    parser.add_argument('--zero-shot-labels', type=str, help='Comma-separated labels (zero-shot mode)')
    parser.add_argument('--zero-shot-model', default='joeddav/xlm-roberta-large-xnli')
    parser.add_argument('--context-window', type=int, default=1, help='Context window size (number of previous/next sentences)')
    args = parser.parse_args()

    build_contextual_sentence_flow(
        processed_dir=args.processed_dir,
        out_csv=args.out_csv,
        method=args.method,
        lexicon_path=args.lexicon_path,
        transformer_model=args.transformer_model,
        zero_shot_labels=args.zero_shot_labels,
        zero_shot_model=args.zero_shot_model,
        context_window=args.context_window,
    )
