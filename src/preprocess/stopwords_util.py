"""stopwords_util.py
Research-oriented Korean stopwords builder.

Modes:
 1) file: user-provided list (one per line)
 2) iso: add stopwordsiso Korean list if available
 3) pos: apply POS-based filtering (requires konlpy) to remove function words
 4) freq: derive high-frequency low-information tokens from corpus (optional)
 5) auto: combination (iso + file + pos(optional) + freq(optional))

Functions:
  load_stopwords(...): returns set[str]

Design:
  - Keep pure functions; heavy operations (POS tagging, freq scan) guarded.
  - Accept dataframe with text column to allow frequency-based filtering.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional, Set, List

try:
    import stopwordsiso as si  # type: ignore
    _HAS_ISO = True
except Exception:
    _HAS_ISO = False

try:
    from konlpy.tag import Okt  # type: ignore
    _HAS_OKT = True
except Exception:
    _HAS_OKT = False

filler_words = {
    # Domain-neutral auxiliary/common verbs & particles often uninformative
    '하다','되다','있다','없다','같다','이다','번째','그것','이것','저것','하다가','하게','하며','그리고','그러나','하지만','또한','그러면서','그러하여','위해'
}

FUNCTION_POS = {'Josa','Eomi','Suffix','Punctuation','Number','Foreign','Alpha'}


def _read_file(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()}


def _tokenize_okt(texts: Iterable[str]) -> List[str]:
    if not _HAS_OKT:
        return []
    okt = Okt()
    tokens: List[str] = []
    for t in texts:
        try:
            for w,pos in okt.pos(t, stem=True):
                if pos in FUNCTION_POS:
                    continue
                if len(w) <= 1:
                    continue
                tokens.append(w)
        except Exception:
            continue
    return tokens


def derive_freq_stopwords(texts: Iterable[str], top_n: int = 100, min_len: int = 1) -> Set[str]:
    from collections import Counter
    # simple whitespace tokenization fallback
    c = Counter()
    for t in texts:
        for tok in str(t).split():
            if len(tok) >= min_len:
                c[tok] += 1
    # pick extremely frequent tokens (heuristic) relative frequency threshold
    if not c:
        return set()
    most = c.most_common(top_n)
    # further prune numeric-like or punctuation-like
    result = {w for w,_ in most if not any(ch.isdigit() for ch in w)}
    return result


def load_stopwords(
    mode: str = 'auto',
    file_path: Optional[str] = None,
    extra: Optional[Iterable[str]] = None,
    use_pos: bool = False,
    freq_corpus: Optional[Iterable[str]] = None,
    freq_top_n: int = 0,
) -> Set[str]:
    """Build a composite stopword set.

    Parameters
    ----------
    mode : {'auto','file','iso','none'}
        auto: combine iso + file + extra (+freq + pos)
    file_path : str
        Custom stopwords file (one token per line)
    extra : Iterable[str]
        Extra tokens always included
    use_pos : bool
        If True and konlpy available, POS filtering candidate filler words added
    freq_corpus : Iterable[str]
        If provided and freq_top_n>0, derive high-frequency tokens
    freq_top_n : int
        Number of top frequent tokens to add as stopwords (heuristic)
    """
    sw: Set[str] = set()
    if mode in {'auto','iso'} and _HAS_ISO:
        try:
            sw |= si.stopwords('ko')
        except Exception:
            pass
    if file_path:
        sw |= _read_file(file_path)
    if extra:
        sw |= {e for e in extra if e}
    # Optional frequency derived
    if freq_corpus and freq_top_n > 0:
        sw |= derive_freq_stopwords(freq_corpus, top_n=freq_top_n)
    # POS-based pruning: we *add* filler/common lemmas to stopwords
    if use_pos and _HAS_OKT:
        sw |= filler_words
    # single-character tokens (Korean/Latin) often noise
    sw |= {w for w in list(sw) if len(w) == 1}
    return sw

__all__ = ['load_stopwords','derive_freq_stopwords']
