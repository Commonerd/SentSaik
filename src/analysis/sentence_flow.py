"""sentence_flow.py
Generate sentence-level sentiment curve.

Similar to paragraph_flow structure but splits more finely at the sentence level to calculate sentiment scores and labels.

Output: results/sentence_flow.csv
Columns:
    article_index, sentence_index, sentence_text, score, label, source, date, title, score_norm

Note:
 - zero-shot: If there are many labels, speed may decrease as the number of sentences increases. Use --limit or sampling if needed.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
import re
import pandas as pd

from .sentiment_model import (
    load_lexicon,
    score_text_lexicon,
    TransformerSentiment,
    zero_shot_classify,
)

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?。!?])\s+')

def _split_sentences_regex(text: str, min_len: int = 4) -> List[str]:
    if not text:
        return []
    raw_parts = _SENT_SPLIT_RE.split(text.strip())
    cleaned = []
    for p in raw_parts:
        ps = p.strip()
        if len(ps) >= min_len:
            cleaned.append(ps)
    return cleaned

def _split_sentences_kss(text: str, min_len: int = 4) -> List[str]:
    try:
        import kss  # type: ignore
    except Exception:
        # fallback to regex if kss not installed
        return _split_sentences_regex(text, min_len=min_len)
    sentences: List[str] = []
    try:
        for s in kss.split_sentences(text):  # type: ignore[attr-defined]
            s_clean = s.strip()
            if len(s_clean) >= min_len:
                sentences.append(s_clean)
    except Exception:
        return _split_sentences_regex(text, min_len=min_len)
    return sentences


_SPLITTERS: Dict[str, Callable[[str, int], List[str]]] = {
    'regex': _split_sentences_regex,
    'kss': _split_sentences_kss,
}


def build_sentence_flow(
    processed_dir: str = 'data/processed',
    out_csv: str = 'results/sentence_flow.csv',
    method: str = 'lexicon',
    lexicon_path: str = 'models/sentiment_lexicon_ko.tsv',
    transformer_model: str = 'brainbert/kcbert-base-sentiment',
    zero_shot_labels: Optional[str] = None,
    zero_shot_model: str = 'joeddav/xlm-roberta-large-xnli',
    limit_articles: Optional[int] = None,
    max_sentences_per_article: Optional[int] = None,
    sentence_splitter: str = 'regex',
) -> Path:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    proc_file = Path(processed_dir) / 'articles_processed.csv'
    if not proc_file.exists():
        raise FileNotFoundError(proc_file)

    df = pd.read_csv(proc_file)
    if limit_articles:
        df = df.head(limit_articles)

    # Prepare models
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

    splitter = _SPLITTERS.get(sentence_splitter, _split_sentences_regex)

    total = len(df)
    for idx, row in df.iterrows():
        article_id = idx
        content = row.get('clean_content') or row.get('content') or ''
        sentences = splitter(str(content), min_len=4)
        if not sentences:
            continue
        if max_sentences_per_article:
            sentences = sentences[:max_sentences_per_article]

        zs_results = None
        if method == 'zero-shot':
            zs_results = zero_shot_classify(sentences, zero_labels_list, model_name=zero_shot_model, batch_size=8)

        for s_idx, sent in enumerate(sentences):
            if method == 'lexicon':
                score = score_text_lexicon(sent, lexicon) if lexicon else 0.0
                label = 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
            elif method in ('transformer','transformer-multi'):
                pred = transformer.predict(sent[:1000]) if transformer else {"score":0.0}
                score = float(pred.get('score',0.0))
                label = pred.get('top_label') or ('positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral'))
            elif method == 'zero-shot':
                zr = zs_results[s_idx]
                score = float(zr.get('score',0.0))
                label = zr.get('top_label')
            else:
                score = 0.0
                label = 'neutral'
            # Log output for sentiment judgment per sentence
            print(f"[문장 감정] article:{article_id} sent_idx:{s_idx} label:{label} score:{score:.4f} text:{sent[:50]}", flush=True)
            rows.append({
                'article_index': article_id,
                'sentence_index': s_idx,
                'sentence_text': sent,
                'score': score,
                'label': label,
                'source': row.get('source'),
                'date': row.get('date'),
                'title': row.get('title'),
            })
        # Output progress percentage
        if (idx + 1) % max(1, total // 100) == 0 or (idx + 1) == total:
            percent = int((idx + 1) / total * 100)
            print(f"[진행률] {idx + 1}/{total} ({percent}%) 완료", flush=True)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df['score_norm'] = out_df.groupby('article_index')['score'].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    out_path = Path(out_csv)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Sentence flow saved -> {out_path}")
    return out_path


if __name__ == '__main__':
    build_sentence_flow(method='lexicon')
