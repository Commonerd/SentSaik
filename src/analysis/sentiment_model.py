from __future__ import annotations
"""sentiment_model.py
Sentiment analysis module (extended)
- Lexicon-based scoring
- Transformer (binary/multi-class) classification
- Multi-class custom label support (model's own labels)
- Zero-shot classification (XNLI-based) with user-specified labels
"""
import os
# suppress transformers and HF logs before import
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
import os
from pathlib import Path
from typing import List, Dict, Optional, Sequence
import math

import pandas as pd

try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline)
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

LEXICON_COLUMNS = ["word", "polarity"]  # polarity: -1, 0, 1

# ---------------- Lexicon -----------------

def load_lexicon(path: str | Path) -> Dict[str, int]:
    lex: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            w, p = parts
            try:
                lex[w] = int(p)
            except ValueError:
                continue
    return lex


def score_text_lexicon(text: str, lexicon: Dict[str, int]) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    scores = [lexicon.get(tok, 0) for tok in tokens]
    if not scores:
        return 0.0
    return sum(scores) / (len(scores) ** 0.5)

# --------------- Transformer (supervised) ---------------

RECOMMENDED_SENTIMENT_MODELS: List[str] = [
    "brainbert/kcbert-base-sentiment",  # widely used Korean sentiment (binary or multi)
    "nlptown/bert-base-multilingual-uncased-sentiment",  # 1-5 stars style
    "snunlp/KR-FinBert-Sentiment",  # domain-specific but stable head
]


def recommend_model(prefer_multiclass: bool = False) -> str:
    """Return a recommended model name.
    prefer_multiclass: if True, return a model with 3-5 level labels if available.
    """
    if prefer_multiclass:
        for m in RECOMMENDED_SENTIMENT_MODELS:
            if 'multilingual-uncased-sentiment' in m:
                return m
    return RECOMMENDED_SENTIMENT_MODELS[0]


class TransformerSentiment:
    def __init__(self, model_name: str = "brainbert/kcbert-base-sentiment", device: Optional[str] = None):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 패키지 또는 모델 로드 불가")
        self.model_name = model_name
        load_errors: List[str] = []
        attempted: List[str] = []
        primary_model = model_name
        fallback_models = [
            # good sentiment heads first
            "brainbert/kcbert-base-sentiment",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "snunlp/KR-FinBert-Sentiment",
            # generic / may lack head
            "klue/bert-base",
            "beomi/KcELECTRA-base",
        ]
        models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]
        self.model = None
        self.tokenizer = None
        for m in models_to_try:
            try:
                attempted.append(m)
                self.tokenizer = AutoTokenizer.from_pretrained(m)
                self.model = AutoModelForSequenceClassification.from_pretrained(m)
                self.model_name = m
                break
            except Exception as e:  # broad: want to try fallback
                load_errors.append(f"{m}: {type(e).__name__}: {e}")
                self.tokenizer = None
                self.model = None
                continue
        if self.model is None or self.tokenizer is None:
            help_msg = (
                "모델 로드 실패. 시도한 모델들: " + ", ".join(attempted) + "\n"
                "에러 요약:\n- " + "\n- ".join(load_errors[:5]) + ("\n..." if len(load_errors) > 5 else "") + "\n\n"
                "해결 가이드:\n"
                "1) 필수 패키지 설치 여부 확인: pip install --upgrade transformers torch protobuf sentencepiece\n"
                "2) 일부 모델은 tiktoken 필요: pip install tiktoken (필요시)\n"
                "3) 네트워크 문제나 Hugging Face 인증 토큰 여부 확인 (프록시/방화벽).\n"
                "4) 다른 모델 이름을 --transformer-model 로 직접 지정 해보세요 (예: klue/bert-base).\n"
            )
            raise RuntimeError(help_msg)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # 일부 모델(token_type_ids 범위 문제) 방어
        if 'token_type_ids' in inputs:
            # kobert issue: sometimes token_type_ids length mismatch isn't typical but keep placeholder
            pass
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            outputs = self.model(**inputs)
        except IndexError as e:
            # token_type embeddings range error or classifier mismatch
            return {"score": 0.0, "top_label": "unknown", "error": f"model_forward_index_error: {e}"}
        logits = outputs.logits
        num_labels = logits.shape[-1]
        if num_labels == 1:
            score = float(logits.squeeze().cpu())
            prob_pos = 1 / (1 + math.exp(-score))
            return {"score": score, "prob_positive": prob_pos}
        # Detect possibly uninitialized classifier (very large std)
        with torch.no_grad():
            w = getattr(self.model, 'classifier', None)
            if w is not None and hasattr(w, 'weight'):
                std_val = float(w.weight.std().cpu())
                if std_val > 0.3 and text.strip():  # heuristic threshold
                    # warn via field
                    pass
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # label names if exist
        id2label = getattr(self.model.config, 'id2label', {i: f'label_{i}' for i in range(num_labels)})
        label_probs = {id2label[i]: float(p) for i, p in enumerate(probs)}
        # Polarity derivation
        if num_labels == 2:
            # assume index 1 is positive
            pos = probs[1]
            score = float(pos * 2 - 1)
        elif num_labels == 5 and any("multilingual-uncased-sentiment" in self.model_name for _ in [0]):
            # nlptown 1..5 stars mapping -> linear scale [-1,1]
            # Probs correspond to stars 1..5
            expected = sum((i + 1) * probs[i] for i in range(5))  # 1..5
            score = (expected - 3) / 2  # center 3 ->0; 1-> -1; 5-> +1
        else:
            neg_indices = [i for i, lab in id2label.items() if any(k in lab.lower() for k in ["1", "neg", "bad", "부정"])]
            pos_indices = [i for i, lab in id2label.items() if any(k in lab.lower() for k in ["5", "pos", "good", "긍정"])]
            mid_indices = [i for i, lab in id2label.items() if any(k in lab.lower() for k in ["3", "neutral", "중립"]) ]
            score = 0.0
            if pos_indices or neg_indices:
                score = sum(probs[i] for i in pos_indices) - sum(probs[i] for i in neg_indices)
            elif mid_indices and len(label_probs) >= 3:
                # crude central weighting
                ordered = sorted(label_probs.items(), key=lambda x: x[0])
                # fallback: treat highest label as positive, lowest as negative
                pos = ordered[-1][1]
                neg = ordered[0][1]
                score = pos - neg
        top_label = max(label_probs.items(), key=lambda x: x[1])[0]
        return {"score": score, "top_label": top_label, "probs": label_probs}

# --------------- Zero-shot (XNLI) ---------------

def zero_shot_classify(texts: Sequence[str], labels: Sequence[str], model_name: str = "joeddav/xlm-roberta-large-xnli", batch_size: int = 4, device: Optional[int] = None) -> List[Dict[str, float]]:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers 설치 필요")
    # device: pipeline uses int (GPU index) or -1 for CPU
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model_name, device=device)
    # Filter out completely empty/whitespace texts (HF pipeline will raise when given only empties)
    cleaned: List[str] = []
    valid_indices: List[int] = []
    for idx, t in enumerate(texts):
        ts = (t if isinstance(t, str) else str(t)).strip()
        if ts:
            cleaned.append(ts)
            valid_indices.append(idx)
    # Prepare results container aligned to original order
    results: List[Optional[Dict[str, float]]] = [None] * len(texts)
    if not labels or len(list(labels)) == 0:
        raise ValueError("zero_shot_classify: labels list is empty after preprocessing")
    if not cleaned:
        # All texts empty – return neutral placeholder probabilities
        placeholder = {l: (1.0 / len(list(labels))) for l in labels}
        for i in range(len(texts)):
            results[i] = {"top_label": next(iter(placeholder)), "probs": placeholder, "score": placeholder[next(iter(placeholder))]}
        return results  # type: ignore

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        if not batch:
            continue
        out = clf(batch, candidate_labels=list(labels), multi_label=False)
        if isinstance(out, dict):  # single sample case
            out = [out]
        for local_idx, r in enumerate(out):
            label_scores = dict(zip(r['labels'], r['scores']))
            top_label = r['labels'][0] if r.get('labels') else next(iter(labels))
            score = label_scores.get(top_label, 0.0)
            orig_index = valid_indices[i + local_idx]
            results[orig_index] = {"top_label": top_label, "probs": label_scores, "score": score}

    # Fill any remaining Nones (should only be from empty strings) with uniform placeholder
    if any(r is None for r in results):
        placeholder = {l: (1.0 / len(list(labels))) for l in labels}
        for idx, r in enumerate(results):
            if r is None:
                results[idx] = {"top_label": next(iter(placeholder)), "probs": placeholder, "score": placeholder[next(iter(placeholder))]}
    return results  # type: ignore

# --------------- Main analyze API ---------------

def analyze_sentiment(
    input_dir: str,
    output_dir: str,
    method: str = "lexicon",
    lexicon_path: str = "models/sentiment_lexicon_ko.tsv",
    transformer_model: str = "skt/kobert-base-v1",
    zero_shot_labels: Optional[str] = None,
    zero_shot_model: str = "joeddav/xlm-roberta-large-xnli",
    limit: Optional[int] = None,
) -> Path:
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    in_path = Path(input_dir)
    proc_file = in_path / "articles_processed.csv" if in_path.is_dir() else in_path
    if not proc_file.exists():
        raise FileNotFoundError(f"Processed file not found: {proc_file}")

    df = pd.read_csv(proc_file)
    if limit:
        df = df.head(limit)

    results: List[Dict[str, object]] = []

    if method == "lexicon":
        if not os.path.exists(lexicon_path):
            raise FileNotFoundError(f"Lexicon not found: {lexicon_path}")
        lexicon = load_lexicon(lexicon_path)
        for _, row in df.iterrows():
            text = row.get("tokenized_sentences") or row.get("clean_content", "")
            score = score_text_lexicon(str(text), lexicon)
            label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
            results.append({
                "source": row.get("source"),
                "date": row.get("date"),
                "title": row.get("title"),
                "score": score,
                "label": label,
                "method": method,
            })
    elif method in ("transformer", "transformer-multi"):
        predictor = TransformerSentiment(model_name=transformer_model)
        for _, row in df.iterrows():
            text = str(row.get("clean_content", ""))[:2000]
            pred = predictor.predict(text)
            score = pred.get("score", 0.0)
            label = pred.get("top_label") or ("positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral"))
            entry = {
                "source": row.get("source"),
                "date": row.get("date"),
                "title": row.get("title"),
                "score": score,
                "label": label,
                "method": method,
            }
            probs = pred.get("probs")
            if probs:
                for k, v in probs.items():
                    entry[f"prob_{k}"] = v
            results.append(entry)
    elif method == "zero-shot":
        if not zero_shot_labels:
            raise ValueError("zero_shot_labels must be provided for zero-shot method")
        labels = [l.strip() for l in zero_shot_labels.split(',') if l.strip()]
        # Robust column fallback without ambiguous Series truth-value checks
        if "clean_content" in df.columns and df["clean_content"].notna().any():
            base_series = df["clean_content"].fillna("")
        elif "original_content" in df.columns and df["original_content"].notna().any():
            base_series = df["original_content"].fillna("")
        elif "content" in df.columns:
            base_series = df["content"].fillna("")
        else:
            raise ValueError("No suitable text column found for zero-shot (expected clean_content/original_content/content)")
        texts = [str(t)[:1000] for t in base_series.tolist()]
        zs_results = zero_shot_classify(texts, labels, model_name=zero_shot_model)
        for row, zr in zip(df.to_dict(orient='records'), zs_results):
            entry = {
                "source": row.get("source"),
                "date": row.get("date"),
                "title": row.get("title"),
                "score": zr.get("score"),  # top label prob as scalar
                "label": zr.get("top_label"),
                "method": method,
            }
            probs = zr.get("probs") or {}
            for k, v in probs.items():
                entry[f"prob_{k}"] = v
            results.append(entry)
    else:
        raise ValueError("method must be one of: lexicon, transformer, transformer-multi, zero-shot")

    out_df = pd.DataFrame(results)
    out_path = Path(output_dir)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Sentiment results saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    analyze_sentiment("data/processed", "results/sentiment.csv", method="lexicon")
