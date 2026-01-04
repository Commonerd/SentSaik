"""clean_text.py
Korean news article preprocessing utility.
- Clean HTML/whitespace/special characters
- Sentence-level splitting (simple heuristic)
- Morphological analysis (optional: KoNLPy Okt)
- Stopword removal

In actual research, extend with custom stopword dictionaries, custom regex, named entity recognition, etc.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Iterable, Dict

import pandas as pd

# --------------------------------------------------------------
# Okt (KoNLPy) loading
# Trust konlpy's internal initialization by not calling startJVM directly.
# To suppress Java 21 native access warnings, set in shell before running:
#   export JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED"
# Here, only attempt import.
# --------------------------------------------------------------
_OKT_AVAILABLE = False
try:
    from konlpy.tag import Okt  # type: ignore
    _OKT_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Okt unavailable -> fallback whitespace tokenization. Reason: {e}")
    Okt = None  # type: ignore

# Simple sentence splitting / regex
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")
_CLEAN_RE = re.compile(r"[^0-9A-Za-z가-힣\s.,!?·-]")

DEFAULT_STOPWORDS = {"으로", "이다", "에서", "그리고", "그러나", "하지만", "또한", "같은"}


def basic_clean(text) -> str:
    """Robust cleaning.

    Accepts any type; non-string (NaN/None/float) coerced to empty string.
    """
    if not isinstance(text, str):
        if pd.isna(text):  # type: ignore[arg-type]
            text = ""
        else:
            text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)  # HTML 태그 제거
    text = _CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str, use_morph: bool = True) -> List[str]:
    if use_morph and _OKT_AVAILABLE and Okt is not None:
        try:
            okt = Okt()
            return okt.morphs(text)
        except Exception as e:
            print(f"[WARN] Okt morphs failed -> fallback whitespace. Reason: {e}")
    return text.split()


def remove_stopwords(tokens: Iterable[str], stopwords: set[str] = DEFAULT_STOPWORDS) -> List[str]:
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def preprocess_row(row: Dict[str, str], use_morph: bool = True) -> Dict[str, str]:
    content = row.get("content", "")
    cleaned = basic_clean(content)
    sentences = sentence_split(cleaned)
    tokenized_sentences = []
    for sent in sentences:
        tokens = tokenize(sent, use_morph=use_morph)
        tokens = remove_stopwords(tokens)
        tokenized_sentences.append(" ".join(tokens))
    row["clean_content"] = cleaned
    row["sentences"] = "\n".join(sentences)
    row["tokenized_sentences"] = "\n".join(tokenized_sentences)
    return row


def preprocess_articles(input_dir: str, output_dir: str, filename_pattern: str = "articles_raw.csv", use_morph: bool = True) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    raw_path = Path(input_dir) / filename_pattern
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    processed_rows = []
    for _, row in df.iterrows():
        processed_rows.append(preprocess_row(row.to_dict(), use_morph=use_morph))

    out_df = pd.DataFrame(processed_rows)
    # 'url' 컬럼이 있으면 제거
    if 'url' in out_df.columns:
        out_df = out_df.drop(columns=['url'])
    out_path = Path(output_dir) / "articles_processed.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved processed data -> {out_path}")
    return out_path


if __name__ == "__main__":
    preprocess_articles("data/raw", "data/processed")
