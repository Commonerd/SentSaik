"""plot_sentiment.py
감정 및 파생 분석 결과 시각화 (Korean + English bilingual charts).

Features / 기능:
1) Sentiment results (`sentiment.csv`)
   - Monthly average sentiment trend / 월별 평균 감정 추이
   - Per-source distribution & mean / 매체별 분포 및 평균
   - Label counts & mean probabilities / 레이블 빈도 및 평균 확률
2) Paragraph flow (`paragraph_flow.csv`)
   - Per-article paragraph sentiment curve / 기사별 문단 감정 흐름
   - Global paragraph score distribution / 전체 문단 점수 분포
3) Topic + sentiment (`topic_sentiment_docs.csv`, `topic_sentiment_summary.csv`)
   - Docs per topic / 토픽별 문서 수
   - Average sentiment per topic / 토픽별 평균 감정
   - Label composition per topic / 토픽별 감정 비율
4) Bias / Framing (`bias_summary.csv`)
   - Bias index, neutral ratio, entropy comparisons / 편향, 중립, 다양성 비교
   - Bias vs mean sentiment bubble plot / Bias-평균감정 버블 산점도
5) Wordcloud (optional)

All plots include bilingual titles & captions for clarity.
Use `plot_all(results_dir="results")` to auto-generate everything.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List
import platform

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_offline_plot
    _PLOTLY = True
except Exception:
    _PLOTLY = False
try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    _HAS_NX = False

sns.set(style="whitegrid")

# Reapply rcParams after seaborn.set (to prevent Arial from being reset)
try:
    import matplotlib
    from matplotlib import font_manager
    system_name = platform.system().lower()
    pref = []
    if 'darwin' in system_name or 'mac' in system_name:
        apple_path = "/System/Library/Fonts/AppleGothic.ttf"
        if os.path.exists(apple_path):
            try:
                font_manager.fontManager.addfont(apple_path)
            except Exception:
                pass
        pref.append("AppleGothic")
    pref.extend(["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"])    
    avail = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    chosen_runtime = None
    for c in pref:
        if c in avail:
            chosen_runtime = c
            break
    if chosen_runtime:
        matplotlib.rcParams.update({
            'font.family': chosen_runtime,
            'axes.unicode_minus': False,
        })
    else:
        print('[WARN] Korean font not found (post seaborn).')
except Exception as e:
    print(f'[WARN] Runtime font application failed: {e}')

# Korean font configuration (macOS default: AppleGothic). If fails, print user environment.
try:
    import matplotlib
    from matplotlib import font_manager
    system_name = platform.system().lower()
    _FONT_CANDIDATES = []
    if 'darwin' in system_name or 'mac' in system_name:
        # macOS: AppleGothic (Prioritize registration if file is directly checked)
        apple_path = "/System/Library/Fonts/AppleGothic.ttf"
        if os.path.exists(apple_path):
            try:
                font_manager.fontManager.addfont(apple_path)
            except Exception:
                pass
        _FONT_CANDIDATES.append("AppleGothic")
    # Common candidates
    _FONT_CANDIDATES.extend([
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
    ])
    available = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    chosen = None
    for cand in _FONT_CANDIDATES:
        if cand in available:
            chosen = cand
            break
    if chosen:
        matplotlib.rc('font', family=chosen)
        plt.rcParams['axes.unicode_minus'] = False
        # print(f"[INFO] Using Korean font: {chosen}")
    else:
        print("[WARN] No preferred Korean font found. Install NanumGothic or Noto Sans CJK KR.")
except Exception as e:
    print(f"[WARN] Korean font configuration skipped: {e}")

# Determine a reusable Korean font family name for network labels
try:
    from matplotlib import font_manager as _fm_label
    _KOREAN_FONT_LABEL = None
    # Try reuse 'chosen' or 'chosen_runtime' if defined
    if 'chosen' in globals() and globals().get('chosen'):
        _KOREAN_FONT_LABEL = globals().get('chosen')
    elif 'chosen_runtime' in globals() and globals().get('chosen_runtime'):
        _KOREAN_FONT_LABEL = globals().get('chosen_runtime')
    if not _KOREAN_FONT_LABEL:
        _cands = ["AppleGothic","NanumGothic","Malgun Gothic","Noto Sans CJK KR"]
        avail_names = {f.name for f in _fm_label.fontManager.ttflist}
        for c in _cands:
            if c in avail_names:
                _KOREAN_FONT_LABEL = c
                break
except Exception:
    _KOREAN_FONT_LABEL = None

try:
    from wordcloud import WordCloud
    _WC = True
except Exception:
    _WC = False

# ---------------- Plain-language caption helper (Korean / English) ----------------
_CAPTION_FONTSIZE = 9

def _add_caption(ko: str, en: str):
    """Add bilingual caption below figure.
    We separate lines by ' / ' for compactness if short; else newline.
    """
    text = f"{ko} / {en}" if len(ko)+len(en) < 140 else f"{ko}\n{en}"
    try:
        plt.figtext(0.01, -0.085, text, ha='left', va='top', fontsize=_CAPTION_FONTSIZE, wrap=True)
    except Exception:
        pass

def _title(ko: str, en: str) -> str:
    """Return bilingual title (Korean newline English)."""
    return f"{ko}\n{en}"


def plot_results(sentiment_csv: str, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(sentiment_csv)
    # ------------------------------------------------------------------
    # Robust date parsing (mixed separators like 1962-08-09 / 1989.10.07 / 1995/07/05)
    # We DO NOT overwrite the original 'date' column (keep raw for reference).
    # Parsed datetime stored in 'date_dt'. All subsequent .dt access should use 'date_dt'.
    # ------------------------------------------------------------------
    has_date = False
    if 'date' in df.columns:
        try:
            raw = (
                df['date']
                .astype(str)
                .str.strip()
                .str.replace('.', '-', regex=False)
                .str.replace('/', '-', regex=False)
            )
            # Remove any trailing non-digit chars (common scraping artefacts)
            raw = raw.str.replace(r'[^0-9\-]', '', regex=True)
            # Heuristic: if length==8 and no dashes -> assume YYYYMMDD insert dashes
            raw = raw.apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if (len(x)==8 and '-' not in x) else x)
            df['date_dt'] = pd.to_datetime(raw, errors='coerce', infer_datetime_format=True)
            if df['date_dt'].notna().any():
                has_date = True
            else:
                print('[WARN] All date parsing failed (sentiment.csv); date-based plots will be skipped.')
        except Exception as e:
            print(f'[WARN] Date parsing exception: {e}')


    # 1. Monthly trend (guarded)
    if has_date:
        ts = df.groupby(df['date_dt'].dt.to_period('M'))['score'].mean().reset_index()
        ts['date'] = ts['date_dt'].dt.to_timestamp()
        plt.figure(figsize=(10,4))
        plt.plot(ts['date'], ts['score'], marker='o')
        plt.title(_title('Monthly Average Sentiment Trend', 'Monthly Average Sentiment Trend'))
        plt.xlabel('Month')
        plt.xticks(rotation=45)
        _add_caption('전체 기사 감정이 시간에 따라 어떻게 이동하는지 보여줍니다.', 'Shows how overall tone shifts over time (-1 negative to +1 positive).')
        plt.tight_layout(rect=[0,0.05,1,1])
        # plt.savefig(Path(out_dir)/'01_sentiment_trend.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 1b. Overall sentiment score histogram + KDE
    if 'score' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['score'], kde=True, bins=30, color='#4C72B0')
        plt.title(_title('전체 감정 점수 분포', 'Overall Sentiment Score Distribution'))
        plt.xlabel('감정 점수 (-1=부정,0=중립,+1=긍정)')
        plt.ylabel('빈도 Frequency')
        _add_caption('기사들의 감정 점수가 어떤 구간에 몰려 있는지.', 'Where article sentiment scores concentrate.')
        plt.tight_layout(rect=[0,0.05,1,1])
        # plt.savefig(Path(out_dir)/'01b_sentiment_hist.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 1c. Per-source sentiment score histogram (grid) - date parsing removed (not needed here)
    if 'score' in df.columns and 'source' in df.columns:
        sources = df['source'].unique().tolist()
        n = len(sources)
        cols = min(3, n)
        rows = int(np.ceil(n/cols))
        plt.figure(figsize=(4*cols, 2.8*rows))
        for i, src in enumerate(sources, start=1):
            plt.subplot(rows, cols, i)
            sub = df[df['source'] == src]
            sns.histplot(sub['score'], kde=True, bins=20, color='#4C72B0')
            plt.title(str(src))
            if i <= (rows-1)*cols:
                plt.xlabel('')
        plt.suptitle(_title('매체별 감정 점수 분포', 'Sentiment Distribution per Source'), y=1.02)
        _add_caption('각 언론사 점수 분포 비교(모양, 치우침).', 'Compare shape & skew of each source.')
        plt.tight_layout(rect=[0,0.05,1,0.98])
        # plt.savefig(Path(out_dir)/'01c_sentiment_hist_by_source.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Per-source distribution
    if 'source' in df.columns:
        try:
            if df['source'].nunique() >= 1 and df['score'].notna().any():
                plt.figure(figsize=(8,5))
                sns.boxplot(data=df, x='source', y='score')
                plt.title(_title('매체별 감정 점수 분포 (박스)', 'Sentiment Distribution by Source (Box)'))
                plt.xlabel('매체 / Source')
                plt.ylabel('감정 점수 (-1~+1) / Sentiment Score')
                _add_caption('박스: 중간 50% 범위, 중앙선: 중앙값. 점수는 -1(부정)~+1(긍정).', 'Box=IQR, line=median, scale -1 (neg) to +1 (pos).')
                plt.tight_layout(rect=[0,0.05,1,1])
                # plt.savefig(Path(out_dir)/'02_sentiment_box_source.png', dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print('[INFO] Skip source boxplot: insufficient variation')
        except Exception as e:
            print(f'[WARN] source boxplot skipped: {e}')

        plt.figure(figsize=(8,5))
        sns.barplot(data=df, x='source', y='score', estimator='mean', errorbar='sd')
        plt.title(_title('매체별 평균 감정 점수', 'Average Sentiment by Source'))
        plt.xlabel('매체 / Source')
        plt.ylabel('평균 점수 (-1~+1) / Mean Score')
        _add_caption('0보다 크면 더 긍정/우호, 작으면 더 부정/비판 경향.', 'Above 0 = more positive tone; below 0 = more negative.')
        plt.tight_layout(rect=[0,0.05,1,1])
        # plt.savefig(Path(out_dir)/'03_sentiment_bar_source.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. 레이블 분포 및 평균 확률 (존재 시)
    if 'label' in df.columns:
        plt.figure(figsize=(6,4))
        lab_counts = df['label'].value_counts().sort_values(ascending=False)
        sns.barplot(x=lab_counts.index, y=lab_counts.values)
        plt.title(_title('감정 레이블 문서 수', 'Document Count by Sentiment Label'))
        plt.xlabel('레이블 / Label')
        plt.ylabel('문서 수 / Count')
        _add_caption('어떤 감정이 더 자주 나타나는지.', 'Which emotions/labels appear most often.')
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.savefig(Path(out_dir)/'01_label_counts.png', dpi=150, bbox_inches='tight')
        plt.close()

        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        if prob_cols:
            mean_probs = df[prob_cols].mean().sort_values(ascending=False)
            plt.figure(figsize=(6,4))
            sns.barplot(x=[c.replace('prob_','') for c in mean_probs.index], y=mean_probs.values)
            plt.title(_title('평균 레이블 확률', 'Mean Label Probability'))
            plt.xlabel('레이블 / Label')
            plt.ylabel('평균 확률 (0~1) / Mean Prob')
            plt.ylim(0,1)
            _add_caption('모델이 해당 감정을 정답으로 본 평균 확률 (신뢰도).', 'Average model confidence for each label.')
            plt.tight_layout(rect=[0,0.05,1,1])
            # (평균 확률 plot은 산출물에서 제외)
            plt.close()

        # Monthly label count trend (multi-line)
        if has_date:
            try:
                month_label = df.groupby([df['date_dt'].dt.to_period('M'), 'label']).size().reset_index(name='count')
                month_label['month'] = month_label['date_dt'].dt.to_timestamp()
                pivot_counts = month_label.pivot(index='month', columns='label', values='count').fillna(0)
                # Order columns by overall frequency
                order_cols = lab_counts.index.tolist()
                pivot_counts = pivot_counts.reindex(columns=[c for c in order_cols if c in pivot_counts.columns])
                plt.figure(figsize=(10,4))
                for col in pivot_counts.columns:
                    plt.plot(pivot_counts.index, pivot_counts[col], marker='o', label=col)
                plt.title(_title('월별 감정 레이블 등장 추이', 'Monthly Trend of Label Counts'))
                plt.xlabel('월 / Month')
                plt.ylabel('문서 수 / Count')
                plt.xticks(rotation=45)
                plt.legend(title='레이블 / Label', bbox_to_anchor=(1.02,1), loc='upper left')
                _add_caption('각 감정 레이블이 월별로 얼마나 등장했는지 비교.', 'Compare how often each label appears per month.')
                plt.tight_layout(rect=[0,0.05,1,1])
                plt.savefig(Path(out_dir)/'02_label_monthly_count_trend.png', dpi=150, bbox_inches='tight')
                plt.close()

                # Monthly mean probability per label (if probability columns exist)
                prob_cols = [c for c in df.columns if c.startswith('prob_')]
                if prob_cols:
                    month_prob = df.groupby(df['date_dt'].dt.to_period('M'))[prob_cols].mean().reset_index()
                    month_prob['month'] = month_prob['date_dt'].dt.to_timestamp()
                    plt.figure(figsize=(10,4))
                    for pc in prob_cols:
                        label_name = pc.replace('prob_','')
                        plt.plot(month_prob['month'], month_prob[pc], marker='o', label=label_name)
                    plt.title(_title('월별 레이블 평균 확률 추이', 'Monthly Mean Label Probability Trend'))
                    plt.xlabel('월 / Month')
                    plt.ylabel('평균 확률 (0~1) / Mean Probability')
                    plt.ylim(0,1)
                    plt.xticks(rotation=45)
                    plt.legend(title='레이블 / Label', bbox_to_anchor=(1.02,1), loc='upper left')
                    _add_caption('시간에 따라 모델이 각 감정을 얼마나 자신 있게 예측했는지.', 'How model confidence for each label changes over time.')
                    plt.tight_layout(rect=[0,0.05,1,1])
                    plt.savefig(Path(out_dir)/'03_label_monthly_prob_trend.png', dpi=150, bbox_inches='tight')
                    plt.close()

                # Monthly label proportional stacked area (ratio)
                try:
                    if pivot_counts.shape[0] >= 1 and pivot_counts.shape[1] >= 1:
                        ratio = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
                        # (비율 stacked area plot은 산출물에서 제외)
                        y = [ratio[c].values for c in ratio.columns]
                        plt.stackplot(x, *y, labels=ratio.columns, alpha=0.85)
                        plt.title(_title('월별 감정 비율 스택 에어리어', 'Monthly Stacked Area of Label Ratios'))
                        plt.xlabel('월 / Month')
                        plt.ylabel('비율 (0~1) / Ratio')
                        plt.xticks(rotation=45)
                        plt.legend(title='레이블 / Label', bbox_to_anchor=(1.02,1), loc='upper left')
                        _add_caption('각 월 기사 중 레이블 비율(면적). 합계=1.', 'Proportional share of each label per month (areas sum to 1).')
                        plt.tight_layout(rect=[0,0.05,1,1])
                        # plt.savefig(Path(out_dir)/'08_label_monthly_ratio_area.png', dpi=150, bbox_inches='tight')
                        plt.close()
                except Exception:
                    pass
            except Exception:
                pass

    # 4. Score vs Word Count scatter (article length influence)
    if 'score' in df.columns:
        # derive length
        if 'clean_content' in df.columns:
            lengths = df['clean_content'].fillna('').astype(str).str.split().apply(len)
        elif 'content' in df.columns:
            lengths = df['content'].fillna('').astype(str).str.split().apply(len)
        else:
            lengths = pd.Series([np.nan]*len(df))
        df_len = df.copy()
        df_len['word_count'] = lengths
        valid = df_len.dropna(subset=['word_count','score'])
        if not valid.empty and valid['word_count'].nunique() > 1:
            plt.figure(figsize=(7,4))
            plt.scatter(valid['word_count'], valid['score'], alpha=0.35, s=20, edgecolors='none')
            # trend line (lowess-like using rolling mean after binning)
            try:
                bins = pd.qcut(valid['word_count'], q=min(20, valid['word_count'].nunique()), duplicates='drop')
                trend = valid.groupby(bins)['score'].mean()
                xc = trend.index.map(lambda iv: (iv.left + iv.right)/2)
                plt.plot(xc, trend.values, color='red', linewidth=2, label='평균 / Mean')
                plt.legend()
            except Exception:
                pass
            plt.title(_title('감정 점수 vs 문서 길이', 'Sentiment Score vs Article Length'))
            plt.xlabel('단어 수 (공백 기준) / Word Count')
            plt.ylabel('감정 점수 (-1~+1) / Sentiment Score')
            _add_caption('기사 길이가 감정 점수에 어떤 패턴을 보이는지 탐색.', 'Explore whether longer articles correlate with tone.')
            plt.tight_layout(rect=[0,0.06,1,1])
            # plt.savefig(Path(out_dir)/'09_score_vs_length_scatter.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"[INFO] Plots saved under {out_dir}")


def generate_wordcloud(processed_file: str, out_path: str = "results/wordcloud.png", label_filter: Optional[str] = None):
    if not _WC:
        print("[WARN] wordcloud 패키지 미설치로 건너뜀")
        return
    df = pd.read_csv(processed_file)
    text_col = 'tokenized_sentences' if 'tokenized_sentences' in df.columns else 'clean_content'
    if label_filter and 'label' in df.columns:
        df = df[df['label'] == label_filter]
    text = '\n'.join(str(x) for x in df[text_col] if isinstance(x, str))

    # WordCloud 폰트: 시스템에 한글 폰트가 있다면 동일하게 사용
    font_path = None
    try:
        from matplotlib import font_manager as _fm
        if 'chosen' in globals() and chosen:
            # chosen 이름으로 파일 경로 탐색
            for f in _fm.fontManager.ttflist:
                if f.name == chosen:
                    font_path = f.fname
                    break
    except Exception:
        pass

    wc = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wc.to_file(out_path)
    print(f"[INFO] Wordcloud saved -> {out_path}")


def generate_label_wordclouds(sentiment_csv: str = 'results/sentiment.csv', out_dir: str = 'results/wordclouds', text_col_priority: Optional[List[str]] = None, min_freq: int = 2, max_words: int = 200, stopwords_path: Optional[str] = None):
    """Generate per-label wordclouds using article text.

    Parameters
    ----------
    sentiment_csv : str
        Path to sentiment.csv (must contain 'label').
    out_dir : str
        Output directory for wordcloud PNGs.
    text_col_priority : list[str]
        Ordered list of candidate text columns (default tries tokenized then clean). First existing one used.
    min_freq : int
        Minimum token frequency to include.
    max_words : int
        Max words in cloud.
    """
    if not _WC:
        print('[INFO] wordcloud 미설치 → per-label wordcloud 생략')
        return
    if text_col_priority is None:
        text_col_priority = ['tokenized_sentences','clean_content','content']
    if not os.path.exists(sentiment_csv):
        print(f'[INFO] sentiment csv 없음: {sentiment_csv}')
        return
    df = pd.read_csv(sentiment_csv)
    if 'label' not in df.columns:
        print('[INFO] label 컬럼 없음 → per-label wordcloud skip')
        return
    text_col = None
    for c in text_col_priority:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # fallback: try processed articles
        proc_path = 'data/processed/articles_processed.csv'
        if os.path.exists(proc_path):
            try:
                proc = pd.read_csv(proc_path)
                # heuristic merge by title if present
                merged = None
                if 'title' in df.columns and 'title' in proc.columns:
                    for c in text_col_priority:
                        if c in proc.columns:
                            merged = pd.merge(df[['title','label']], proc[['title', c]], on='title', how='left')
                            text_col = c
                            break
                elif 'id' in df.columns and 'id' in proc.columns:
                    for c in text_col_priority:
                        if c in proc.columns:
                            merged = pd.merge(df[['id','label']], proc[['id', c]], on='id', how='left')
                            text_col = c
                            break
                if merged is not None and text_col is not None:
                    df = merged
                    print(f'[INFO] Wordcloud fallback: merged processed column {text_col}')
                else:
                    print('[INFO] 사용 가능한 텍스트 컬럼 & fallback 매칭 실패')
                    return
            except Exception as e:
                print(f'[WARN] wordcloud fallback 실패: {e}')
                return
        else:
            print('[INFO] 사용 가능한 텍스트 컬럼 없음')
            return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # font path reuse
    font_path = None
    try:
        from matplotlib import font_manager as _fm
        if 'chosen' in globals() and chosen:
            for f in _fm.fontManager.ttflist:
                if f.name == chosen:
                    font_path = f.fname
                    break
    except Exception:
        pass
    # load stopwords if provided
    stopwords = set()
    if stopwords_path and os.path.exists(stopwords_path):
        try:
            stopwords = {l.strip() for l in open(stopwords_path, encoding='utf-8') if l.strip()}
            print(f'[INFO] Loaded {len(stopwords)} stopwords for wordcloud')
        except Exception as e:
            print(f'[WARN] stopwords load 실패: {e}')
    # simple token splitting (already tokenized_sentences may have space separated)
    for label, g in df.groupby('label'):
        texts = g[text_col].dropna().astype(str)
        if texts.empty:
            continue
        all_text = '\n'.join(texts)
        # frequency filter via process: naive split
        tokens = [t for t in all_text.split() if len(t) > 1 and t not in stopwords]
        from collections import Counter
        ctr = Counter(tokens)
        filtered_tokens = [tok for tok, cnt in ctr.items() if cnt >= min_freq]
        if not filtered_tokens:
            print(f'[INFO] 라벨 {label}: 기준 빈도 충족 토큰 없음')
            continue
        wc = WordCloud(font_path=font_path, width=900, height=500, background_color='white', max_words=max_words)
        wc.generate(' '.join(filtered_tokens))
        safe = str(label).replace('/', '_')
        fp = out_path / f'wordcloud_{safe}.png'
        wc.to_file(str(fp))
        print(f'[INFO] Label wordcloud -> {fp}')



if __name__ == "__main__":
    plot_results("results/sentiment.csv")

# ---------------------------------------------------------------------------
# 추가 고급 시각화
# ---------------------------------------------------------------------------

def _ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def plot_paragraph_flow(paragraph_csv: str = "results/paragraph_flow.csv", out_dir: str = "results/plots_paragraph", max_articles: int = 12):
    if not os.path.exists(paragraph_csv):
        print(f"[INFO] paragraph flow csv 미존재: {paragraph_csv}")
        return
    out_dir = _ensure_dir(out_dir)
    df = pd.read_csv(paragraph_csv)
    if df.empty:
        print("[INFO] paragraph_flow.csv empty")
        return
    # 기사별 곡선 (score_norm 우선, 없으면 score)
    value_col = 'score_norm' if 'score_norm' in df.columns else 'score'
    # 가장 문단 수 많은 기사 상위 max_articles
    art_sizes = df.groupby('article_index')['paragraph_index'].max().sort_values(ascending=False)
    target_articles = art_sizes.head(max_articles).index.tolist()
    for order, aid in enumerate(target_articles, start=1):
        sub = df[df['article_index'] == aid].sort_values('paragraph_index')
        plt.figure(figsize=(6,3))
        plt.plot(sub['paragraph_index'], sub[value_col], marker='o')
        ttl = str(sub['title'].iloc[0]) if 'title' in sub.columns else f'기사 {aid}'
        short = (ttl[:28] + '…') if len(ttl) > 30 else ttl
        plt.title(_title(f'[{order}] 문단 감정 흐름: {short}', f'Paragraph Sentiment Flow: Article {aid}'))
        plt.xlabel('문단 인덱스 / Paragraph')
        plt.ylabel('정규화 점수(z)' if value_col=='score_norm' else '점수 (-1~+1)')
        _add_caption('문단별 감정 기복(위=긍정, 아래=부정). z는 기사 내부 상대 비교.', 'Per-paragraph trajectory (up=positive, down=negative). z = within-article normalized.')
        plt.tight_layout(rect=[0,0.07,1,1])
        safe_name = f"{order:02d}_paragraph_flow_article{aid}.png"
        # plt.savefig(out_dir / safe_name, dpi=140, bbox_inches='tight')
        plt.close()
    # 전체 분포
    plt.figure(figsize=(6,4))
    sns.violinplot(data=df, x=None, y=value_col)
    plt.title(_title('전체 문단 감정 점수 분포', 'Distribution of Paragraph Sentiment Scores'))
    _add_caption('폭이 넓은 부분은 해당 점수 문단이 많음.', 'Wider sections indicate more paragraphs at that score.')
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig(out_dir / '00_paragraph_score_violin.png', dpi=140, bbox_inches='tight')
    plt.close()

    try:
        plt.figure(figsize=(6,4))
        if 'label' in df.columns and df['label'].nunique() > 1:
            sns.boxplot(data=df, x='label', y=value_col)
            ttl = _title('문단 점수 (레이블별)', 'Paragraph Scores (by Label)')
        else:
            sns.boxplot(data=df, y=value_col)
            ttl = _title('문단 점수 분포', 'Paragraph Score Distribution')
        plt.title(ttl)
        _add_caption('문단 레벨 감정 분포 비교.', 'Comparison of paragraph-level sentiment distributions.')
        plt.tight_layout(rect=[0,0.07,1,1])
        # plt.savefig(out_dir / '00_paragraph_score_box.png', dpi=140, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'[WARN] paragraph boxplot skipped: {e}')
    print(f"[INFO] Paragraph flow plots -> {out_dir}")


def plot_sentence_flow(sentence_csv: str = "results/sentence_flow.csv", out_dir: str = "results/plots_sentence", max_articles: int = 15):
    """문장 단위 감정 흐름 및 분포 플롯 생성.

    Parameters
    ----------
    sentence_csv : str
        results/sentence_flow.csv 경로 (build_sentence_flow 출력)
    out_dir : str
        플롯 저장 경로
    max_articles : int
        개별 곡선 플롯 최대 기사 수 (문장 수 많은 순)
    """
    if not os.path.exists(sentence_csv):
        print(f"[INFO] sentence flow csv 미존재: {sentence_csv}")
        return
    out_dir = _ensure_dir(out_dir)
    df = pd.read_csv(sentence_csv)
    if df.empty:
        print('[INFO] sentence_flow.csv empty')
        return
    value_col = 'score_norm' if 'score_norm' in df.columns else 'score'
    # 문장 수 많은 기사 순 정렬 (sentence_index 최대치 사용)
    art_sizes = df.groupby('article_index')['sentence_index'].max().sort_values(ascending=False)
    target_articles = art_sizes.head(max_articles).index.tolist()
    for order, aid in enumerate(target_articles, start=1):
        sub = df[df['article_index']==aid].sort_values('sentence_index')
        plt.figure(figsize=(6,3))
        plt.plot(sub['sentence_index'], sub[value_col], marker='o', linewidth=1.2)
        ttl = str(sub['title'].iloc[0]) if 'title' in sub.columns else f'기사 {aid}'
        short = (ttl[:32] + '…') if len(ttl) > 34 else ttl
        plt.title(_title(f'[{order}] 문장 감정 흐름: {short}', f'Sentence Sentiment Flow: Article {aid}'))
        plt.xlabel('문장 인덱스 / Sentence Index')
        plt.ylabel('정규화 점수(z)' if value_col=='score_norm' else '점수 (-1~+1)')
        _add_caption('문장 단위 미세 감정 기복 (z=기사 내부 표준화).', 'Fine-grained per-sentence sentiment (z=within-article).')
        plt.tight_layout(rect=[0,0.07,1,1])
        safe_name = f"{order:02d}_sentence_flow_article{aid}.png"
        plt.savefig(Path(out_dir)/safe_name, dpi=140, bbox_inches='tight')
        plt.close()
    # 전체 분포 (violin + box by label if available)
    plt.figure(figsize=(6,4))
    sns.violinplot(data=df, y=value_col, inner='quartile')
    plt.title(_title('전체 문장 감정 분포', 'Sentence Sentiment Distribution'))
    _add_caption('폭이 넓은 영역 = 해당 점수 문장 많음.', 'Wider violin sections = many sentences at score.')
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig(Path(out_dir)/'00_sentence_score_violin.png', dpi=140, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6,4))
    if 'label' in df.columns:
        sns.boxplot(data=df, x='label', y=value_col)
        plt.title(_title('문장 점수 (레이블별)', 'Sentence Scores (by Label)'))
    else:
        sns.boxplot(data=df, y=value_col)
        plt.title(_title('문장 점수 분포', 'Sentence Score Distribution'))
    _add_caption('문장 레벨 감정 점수 범위 비교.', 'Distribution of sentence-level sentiment.')
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig(Path(out_dir)/'00_sentence_score_box.png', dpi=140, bbox_inches='tight')
    plt.close()

    # 문장 길이 vs 점수 (토큰수 기준)
    if 'sentence_text' in df.columns and 'score' in df.columns:
        tmp = df.copy()
        tmp['length'] = tmp['sentence_text'].fillna('').astype(str).str.split().apply(len)
        valid = tmp[tmp['length']>0]
        if not valid.empty:
            plt.figure(figsize=(6.2,4))
            plt.scatter(valid['length'], valid['score'], alpha=0.3, s=14, edgecolors='none')
            plt.title(_title('문장 길이 vs 감정 점수', 'Sentence Length vs Sentiment'))
            plt.xlabel('토큰 수 / Tokens')
            plt.ylabel('감정 점수 (-1~+1)')
            _add_caption('문장 길이와 점수 상관 경향 탐색.', 'Explore correlation between sentence length and tone.')
            plt.tight_layout(rect=[0,0.07,1,1])
            # plt.savefig(Path(out_dir)/'00_sentence_length_scatter.png', dpi=140, bbox_inches='tight')
            plt.close()
    print(f"[INFO] Sentence flow plots -> {out_dir}")


def plot_topic_results(
    topic_docs_csv: str = 'results/topic_sentiment_docs.csv',
    topic_summary_csv: str = 'results/topic_sentiment_summary.csv',
    out_dir: str = 'results/plots_topic',
    max_topics: int = 30,
):
    if not (os.path.exists(topic_docs_csv) and os.path.exists(topic_summary_csv)):
        print('[INFO] topic csv 미존재: skip')
        return
    out_dir = _ensure_dir(out_dir)
    docs = pd.read_csv(topic_docs_csv)
    summ = pd.read_csv(topic_summary_csv)
    if summ.empty:
        print('[INFO] topic summary empty (min_docs 필터로 제거)')
        return
    # topic 이름 준비
    if 'Name' in summ.columns:
        summ['topic_label'] = summ.apply(lambda r: f"{r['topic']}:{str(r['Name'])[:40]}", axis=1)
    else:
        summ['topic_label'] = summ['topic'].astype(str)
    summ = summ.sort_values('n_docs', ascending=False).head(max_topics)

    plt.figure(figsize=(max(8, len(summ)*0.6), 4))
    sns.barplot(data=summ, x='topic_label', y='n_docs')
    plt.xticks(rotation=60, ha='right')
    plt.title(_title('토픽별 문서 수', 'Documents per Topic'))
    _add_caption('많을수록 언론 관심이 높은 주제.', 'More docs = higher media attention.')
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig(Path(out_dir)/'topic_docs_count.png', dpi=140, bbox_inches='tight')
    plt.close()

    if 'avg_score' in summ.columns:
        plt.figure(figsize=(max(8, len(summ)*0.6), 4))
        sns.barplot(data=summ, x='topic_label', y='avg_score')
        plt.xticks(rotation=60, ha='right')
        plt.title(_title('토픽별 평균 감정 점수', 'Average Sentiment by Topic'))
        _add_caption('0보다 높으면 긍정/우호적 어조가 많은 주제.', 'Above 0 implies more positive/empathetic tone.')
        plt.tight_layout(rect=[0,0.07,1,1])
        # plt.savefig(Path(out_dir)/'topic_avg_score.png', dpi=140, bbox_inches='tight')
        plt.close()

    # 레이블 비율 스택드 바 (docs 기준)
    if 'label' in docs.columns:
        label_pivot = docs.pivot_table(index='topic', columns='label', values='score', aggfunc='count', fill_value=0)
        # 정규화 비율
        label_ratio = label_pivot.div(label_pivot.sum(axis=1), axis=0)
        # topic 정렬 동일하게
        label_ratio = label_ratio.loc[summ['topic']]
        label_ratio.index = summ['topic_label']
        plt.figure(figsize=(max(8, len(summ)*0.6), 5))
        bottom = np.zeros(len(label_ratio))
        for col in label_ratio.columns:
            vals = label_ratio[col].values
            plt.bar(label_ratio.index, vals, bottom=bottom, label=col)
            bottom += vals
        plt.xticks(rotation=60, ha='right')
        plt.ylabel('비율')
        plt.title(_title('토픽별 레이블 구성 비율', 'Label Composition by Topic'))
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        _add_caption('주제 내 감정 비율 비교.', 'Distribution of labels within each topic.')
        plt.tight_layout(rect=[0,0.08,1,1])
        # plt.savefig(Path(out_dir)/'topic_label_composition.png', dpi=140, bbox_inches='tight')
        plt.close()

    # Topic vs Month heatmap (document counts)
    if 'date' in docs.columns:
        try:
            docs['date'] = pd.to_datetime(docs['date'])
            docs_f = docs[docs['topic'].isin(summ['topic'])].copy()
            docs_f['month'] = docs_f['date'].dt.to_period('M').dt.to_timestamp()
            heat = docs_f.pivot_table(index='topic', columns='month', values='score', aggfunc='count', fill_value=0)
            # reorder topics as in summary
            heat = heat.loc[summ['topic']]
            # Replace row index with topic_label
            mapper = dict(zip(summ['topic'], summ['topic_label']))
            heat.index = [mapper[t] for t in heat.index]
            plt.figure(figsize=(max(8, len(heat.columns)*0.5), max(6, len(heat)*0.4)))
            sns.heatmap(heat, cmap='Blues', linewidths=.5, annot=False)
            plt.title(_title('토픽별 월별 문서 수 히트맵', 'Topic vs Month Document Count Heatmap'))
            plt.xlabel('월 / Month')
            plt.ylabel('토픽 / Topic')
            _add_caption('어떤 주제가 언제 집중 보도됐는지 (짙을수록 문서 많음).', 'When each topic was most covered (darker = more docs).')
            plt.tight_layout(rect=[0,0.06,1,1])
            # plt.savefig(Path(out_dir)/'topic_month_heatmap.png', dpi=140, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    print(f"[INFO] Topic plots -> {out_dir}")


def plot_bias(bias_csv: str = 'results/bias_summary.csv', out_dir: str = 'results/plots_bias'):
    if not os.path.exists(bias_csv):
        print('[INFO] bias csv 미존재: skip')
        return
    out_dir = _ensure_dir(out_dir)
    df = pd.read_csv(bias_csv)
    if df.empty:
        print('[INFO] bias_summary.csv empty')
        return
    # bias_index
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x='source', y='bias_index')
    plt.title(_title('매체별 Bias Index', 'Bias Index by Source'))
    _add_caption('Bias Index=(긍정-부정)/(긍정+부정). 0보다 크면 긍정 기사 비중 ↑', 'Bias Index=(pos-neg)/(pos+neg). >0 means more positive coverage.')
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig(Path(out_dir)/'bias_index.png', dpi=140, bbox_inches='tight')
    plt.close()

    # neutral_ratio
    if 'neutral_ratio' in df.columns:
        plt.figure(figsize=(6,4))
        sns.barplot(data=df, x='source', y='neutral_ratio')
        plt.title(_title('매체별 중립 문서 비율', 'Neutral Ratio by Source'))
        _add_caption('중립(평이) 톤 기사 비중.', 'Share of neutral-toned articles.')
        plt.tight_layout(rect=[0,0.07,1,1])
        # plt.savefig(Path(out_dir)/'neutral_ratio.png', dpi=140, bbox_inches='tight')
        plt.close()

    # entropy
    if 'entropy_labels' in df.columns:
        plt.figure(figsize=(6,4))
        sns.barplot(data=df, x='source', y='entropy_labels')
        plt.title(_title('매체별 레이블 엔트로피', 'Label Entropy by Source'))
        _add_caption('감정 다양성 지표 (높을수록 여러 감정 고르게).', 'Diversity of emotions (higher = more varied).')
        plt.tight_layout(rect=[0,0.07,1,1])
        # plt.savefig(Path(out_dir)/'entropy_labels.png', dpi=140, bbox_inches='tight')
        plt.close()

    # scatter avg_score vs bias_index
    if {'avg_score','bias_index','n_docs'} <= set(df.columns):
        plt.figure(figsize=(6,4))
        sizes = (df['n_docs'] / df['n_docs'].max()) * 800 + 50
        plt.scatter(df['avg_score'], df['bias_index'], s=sizes, alpha=0.7)
        for _, r in df.iterrows():
            plt.text(r['avg_score'], r['bias_index'], r['source'], fontsize=9, ha='center', va='center')
        plt.xlabel('평균 감정 점수')
        plt.ylabel('Bias Index')
        plt.title(_title('Bias vs 평균 감정 (버블=문서 수)', 'Bias vs Mean Sentiment (bubble=docs)'))
        _add_caption('우상향: 긍정 수준 & 긍정 편향 모두 높음.', 'Upper-right: both positive level and bias high.')
        plt.tight_layout(rect=[0,0.08,1,1])
        # plt.savefig(Path(out_dir)/'bias_vs_avg.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"[INFO] Bias plots -> {out_dir}")


def plot_sentiment_transition_network(edges_csv: str = 'results/network_paragraph/sentiment_transition_edges.csv',
                                      nodes_csv: str = 'results/network_paragraph/sentiment_transition_nodes.csv',
                                      out_dir: str = 'results/plots_network_paragraph',
                                      highlight_top_k: int = 0):
    """Plot sentiment transition directed network using spring layout.

    Creates two figures:
      1) Weighted directed network (edge width ~ count, color by prob)
      2) Transition probability matrix heatmap (label->label)
    """
    if not _HAS_NX:
        print('[INFO] networkx 미설치: transition network plot skip')
        return
    if not (os.path.exists(edges_csv) and os.path.exists(nodes_csv)):
        print('[INFO] transition edges/nodes csv 미존재 skip')
        return
    out_p = _ensure_dir(out_dir)
    edges = pd.read_csv(edges_csv)
    nodes = pd.read_csv(nodes_csv)
    if edges.empty or nodes.empty:
        print('[INFO] transition network empty')
        return
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r.get('prob', 0.0))
    # Node attributes
    pagerank_map = {r['label']: r.get('pagerank', 0.0) for _, r in nodes.iterrows()} if 'pagerank' in nodes.columns else {}
    # Layout
    pos = nx.spring_layout(G, weight='weight', seed=42)
    fig, ax = plt.subplots(figsize=(6,6))
    # Edge widths
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights)
    width_scaled = [1.0 + 4.0*(w/max_w) for w in weights]
    # Edge colors by probability
    probs = [G[u][v].get('prob',0.0) for u,v in G.edges()]
    cmap = plt.cm.Blues
    edge_colors = [cmap(min(0.95,p)) for p in probs]
    # Optional highlight of top-k outgoing edges per node
    highlight_edges = set()
    if highlight_top_k > 0:
        for src in edges['label'].unique():
            sub_top = edges[edges['label']==src].sort_values('count', ascending=False).head(highlight_top_k)
            for _, row in sub_top.iterrows():
                highlight_edges.add((row['label'], row['next_label']))
    draw_colors = []
    alphas = []
    for (u,v), base_col in zip(G.edges(), edge_colors):
        if highlight_edges:
            if (u,v) in highlight_edges:
                draw_colors.append(base_col)
                alphas.append(1.0)
            else:
                # dim non-highlight
                dim = (*base_col[:3], 0.25)
                draw_colors.append(dim)
                alphas.append(0.25)
        else:
            draw_colors.append(base_col)
            alphas.append(0.85)
    # Increase arrow size for clearer directionality
    nx.draw_networkx_edges(G, pos, width=width_scaled, edge_color=draw_colors, arrows=True, arrowsize=20, ax=ax)
    # Node size by (in+out) degree weight or pagerank fallback
    if pagerank_map:
        sizes = [400 + 2500*pagerank_map.get(n,0) for n in G.nodes()]
    else:
        sizes = [400 + 30*G.degree(n, weight='weight') for n in G.nodes()]
    # Node colors by sentiment label (user requested: 긍정=blue, 부정=red, 동정=yellow)
    _label_color_map = {
        '긍정': (31/255,119/255,180/255,0.78),   # matplotlib default blue
        '부정': (214/255,39/255,40/255,0.78),    # matplotlib default red
        '동정': (255/255,219/255,88/255,0.85),   # warm yellow (slightly higher alpha for visibility)
    }
    node_colors = [_label_color_map.get(n, (0.7,0.7,0.7,0.6)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='#333', linewidths=0.9, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax, font_family=_KOREAN_FONT_LABEL if _KOREAN_FONT_LABEL else 'sans-serif')
    # Edge annotations (count/prob)
    # Bidirectional edge label placement: if reverse edge exists offset one up, one down
    for (u,v) in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        if u == v:
            # Improved self-loop label: dynamic offset above node, avoid inside overlap
            data = G[u][v]
            import math
            loop_off = 0.10 + 0.02 * (1 / math.sqrt(len(G.nodes()) or 1))
            ax.text(x0, y0 + loop_off, f"loop {data.get('weight',0)}/{data.get('prob',0):.2f}", fontsize=8,
                    ha='center', va='center', color='black', alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#666', alpha=0.55))
            continue
        # Directional midpoint slightly biased toward target to emphasize direction.
        dir_bias = 0.58  # >0.5 pulls label toward target node
        mid_x = x0 + (x1 - x0) * dir_bias
        mid_y = y0 + (y1 - y0) * dir_bias
        dx = x1 - x0; dy = y1 - y0
        import math
        length = math.hypot(dx, dy) or 1.0
        perp_x, perp_y = -dy/length, dx/length  # perpendicular unit vector
        has_reverse = G.has_edge(v,u)
        # Adaptive offset based on edge count to reduce overlaps
        offset_scale = 0.035 + 0.018 * (len(G.edges())/10)**0.5
        # Stable ordering for consistent above/below assignment
        off = offset_scale if (u < v or not has_reverse) else -offset_scale
        if has_reverse and (v < u):  # ensure opposite sign when iterating the paired edge second
            off = -off
        label_x = mid_x + perp_x * off
        label_y = mid_y + perp_y * off
        data = G[u][v]
        ax.text(label_x, label_y, f"{data.get('weight',0)}/{data.get('prob',0):.2f}", fontsize=8,
                ha='center', va='center', color='darkslategray', alpha=0.92,
                bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none', alpha=0.58))
    # Legend / explanation for edge label format
    ax.text(1.02, 0.02, 'edge label: count/prob\n(방향: 화살표)\ncount=전이 횟수\nprob=P(next|current)',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.8))
    # Node color legend (manual) – small patch in empty corner
    legend_txt = '노드 색상 / Node colors:\n긍정=파랑  부정=빨강  동정=노랑'
    ax.text(1.02, 0.55, legend_txt, transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.8))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('전이 확률 / Transition Prob')
    ax.set_title(_title('감정 전이 네트워크', 'Sentiment Transition Network'))
    _add_caption('문단 순서에서 감정 레이블 전이(두께=횟수, 색=확률).', 'Paragraph-level label transitions (width=count, color=prob).')
    fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(out_p/'sentiment_transition_network.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Probability matrix heatmap
    mat = edges.pivot_table(index='label', columns='next_label', values='prob', fill_value=0)
    plt.figure(figsize=(4.5,4))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='Blues')
    plt.title(_title('감정 전이 확률 행렬', 'Transition Probability Matrix'))
    _add_caption('행: 현재 감정, 열: 다음 감정. 값=전이 확률.', 'Row=current label, col=next label, value=probability.')
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(out_p/'sentiment_transition_matrix.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Transition network plots -> {out_p}')

    # Interactive version (Plotly) with same color/label logic
    if _PLOTLY and os.environ.get('INTERACTIVE','0') in {'1','true','True'}:
        try:
            import plotly.graph_objects as go
            inter_dir = out_p / 'interactive'
            inter_dir.mkdir(exist_ok=True)
            # Reuse positions; convert to list ordering
            node_list = list(G.nodes())
            x_vals = [pos[n][0] for n in node_list]
            y_vals = [pos[n][1] for n in node_list]
            color_map_hex = {
                '긍정': 'rgba(31,119,180,0.78)',
                '부정': 'rgba(214,39,40,0.78)',
                '동정': 'rgba(255,219,88,0.85)'
            }
            node_colors_hex = [color_map_hex.get(n, 'rgba(180,180,180,0.6)') for n in node_list]
            # Node sizes scaled similarly (use same sizes list)
            size_map = {n:s for n,s in zip(G.nodes(), sizes)}
            node_sizes = [size_map[n]*0.06 for n in node_list]  # scale down for Plotly marker size
            # Edge segments
            edge_traces = []
            label_annotations = []
            import math
            total_edges = len(G.edges())
            base_offset = 0.035 + 0.018 * (total_edges/10)**0.5
            for (u,v) in G.edges():
                x0,y0 = pos[u]; x1,y1 = pos[v]
                data = G[u][v]
                if u == v:
                    loop_off = 0.10 + 0.02 * (1 / math.sqrt(len(G.nodes()) or 1))
                    label_annotations.append(dict(
                        x=x0, y=y0 + loop_off, text=f"loop {data.get('weight',0)}/{data.get('prob',0):.2f}",
                        showarrow=False, font=dict(size=10,color='black'),
                        bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(80,80,80,0.4)', borderwidth=0.5
                    ))
                    continue
                dx = x1 - x0; dy = y1 - y0
                length = math.hypot(dx, dy) or 1.0
                ux, uy = dx/length, dy/length
                # Shorten line so arrowhead (triangle) can sit at target
                shrink = 0.04
                ex1 = x0 + ux * (1 - shrink) * length
                ey1 = y0 + uy * (1 - shrink) * length
                # Basic line for edge
                edge_traces.append(go.Scatter(x=[x0, ex1], y=[y0, ey1], mode='lines',
                                              line=dict(color='rgba(120,120,120,0.55)', width=1.5),
                                              hoverinfo='skip', showlegend=False))
                # Arrowhead as small filled triangle
                ah_size = 0.02
                left_x = x1 - ux*shrink*length + (-uy)*ah_size
                left_y = y1 - uy*shrink*length + (ux)*ah_size
                right_x = x1 - ux*shrink*length + (uy)*ah_size
                right_y = y1 - uy*shrink*length + (-ux)*ah_size
                edge_traces.append(go.Scatter(x=[x1, left_x, right_x, x1], y=[y1, left_y, right_y, y1],
                                              mode='lines', fill='toself', line=dict(color='rgba(120,120,120,0.55)', width=1),
                                              hoverinfo='skip', showlegend=False))
                # Directional biased midpoint + perpendicular offset
                dir_bias = 0.58
                mid_x = x0 + dx * dir_bias
                mid_y = y0 + dy * dir_bias
                perp_x, perp_y = -uy, ux
                has_reverse = G.has_edge(v,u)
                off = base_offset if (u < v or not has_reverse) else -base_offset
                if has_reverse and (v < u):
                    off = -off
                label_x = mid_x + perp_x * off
                label_y = mid_y + perp_y * off
                label_annotations.append(dict(
                    x=label_x, y=label_y,
                    text=f"{data.get('weight',0)}/{data.get('prob',0):.2f}",
                    showarrow=False,
                    font=dict(size=9,color='darkslategray'),
                    bgcolor='rgba(255,255,255,0.65)',
                    bordercolor='rgba(120,120,120,0.3)', borderwidth=0.5
                ))
            node_trace = go.Scatter(
                x=x_vals, y=y_vals, mode='markers+text',
                text=node_list, textposition='middle center',
                marker=dict(size=node_sizes, color=node_colors_hex, line=dict(color='rgba(50,50,50,0.6)', width=1)),
                hoverinfo='text',
                showlegend=False
            )
            layout = go.Layout(
                title='감정 전이 네트워크 (Interactive)',
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                annotations=label_annotations + [
                    dict(x=1.02, y=0.02, xref='paper', yref='paper', showarrow=False, align='left',
                         text='edge: count/prob<br>loop: 자기전이',
                         font=dict(size=9), bgcolor='rgba(255,255,255,0.85)',
                         bordercolor='rgba(120,120,120,0.4)', borderwidth=0.5),
                    dict(x=1.02, y=0.60, xref='paper', yref='paper', showarrow=False, align='left',
                         text='노드색:<br>긍정=파랑<br>부정=빨강<br>동정=노랑',
                         font=dict(size=9), bgcolor='rgba(255,255,255,0.85)',
                         bordercolor='rgba(120,120,120,0.4)', borderwidth=0.5)
                ],
                margin=dict(l=10,r=180,t=60,b=10),
                paper_bgcolor='white', plot_bgcolor='white'
            )
            fig_i = go.Figure(data=edge_traces + [node_trace], layout=layout)
            fig_i.write_html(inter_dir/'sentiment_transition_network_interactive.html')
        except Exception as e:  # pragma: no cover
            print('[WARN] Interactive sentiment network failed:', e)


def plot_sentence_transition_network(edges_csv: str = 'results/network_sentence/sentiment_transition_edges.csv',
                                     nodes_csv: str = 'results/network_sentence/sentiment_transition_nodes.csv',
                                     out_dir: str = 'results/plots_network_sentence',
                                     highlight_top_k: int = 0):
    """
    Sentence-level sentiment transition network plotting (same logic as paragraph).
    Note: x/y axes have no semantic meaning; node positions are determined by spring layout for visual clarity only.
    """
    if not _HAS_NX:
        return
    if not (os.path.exists(edges_csv) and os.path.exists(nodes_csv)):
        return
    out_p = _ensure_dir(out_dir)
    edges = pd.read_csv(edges_csv)
    nodes = pd.read_csv(nodes_csv)
    if edges.empty or nodes.empty:
        return
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r.get('prob',0.0))
    # Fixed semantic positions for clarity (Negative left, Sympathy top, Positive right)
    semantic_pos = {
        '부정': (-1.0, 0.0),   # Negative
        '동정': (0.0, 1.2),    # Sympathy
        '긍정': (1.0, 0.0),    # Positive
    }
    # Fallback to spring layout for any unexpected labels
    spring_pos = nx.spring_layout(G, weight='weight', seed=24)
    pos = {n: semantic_pos.get(n, spring_pos.get(n, (0.0,0.0))) for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(6,6))
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights)
    width_scaled = [1.0 + 4.0*(w/max_w) for w in weights]
    probs = [G[u][v].get('prob',0.0) for u,v in G.edges()]
    cmap = plt.cm.Greens
    # Use base green colormap but modulate alpha by probability for clearer intensity differences
    base_cols = [cmap(min(0.95,p)) for p in probs]
    edge_colors = []
    for (rgba, p) in zip(base_cols, probs):
        edge_colors.append((rgba[0], rgba[1], rgba[2], 0.25 + 0.7*min(1.0, p)))
    highlight_edges = set()
    if highlight_top_k > 0:
        for src in edges['label'].unique():
            sub_top = edges[edges['label']==src].sort_values('count', ascending=False).head(highlight_top_k)
            for _, row in sub_top.iterrows():
                highlight_edges.add((row['label'], row['next_label']))
    draw_colors = []
    for (u,v), base_col in zip(G.edges(), edge_colors):
        if highlight_edges:
            if (u,v) in highlight_edges:
                draw_colors.append(base_col)
            else:
                dim = (*base_col[:3], 0.25)
                draw_colors.append(dim)
        else:
            draw_colors.append(base_col)
    # --- Custom edge drawing with curved arcs for bidirectional pairs ---
    # For each unordered pair {u,v}, if both directions exist, draw one arc curving upward and the other downward.
    # We use matplotlib.patches.FancyArrowPatch with connectionstyle 'arc3,rad=±r'.
    import math
    from matplotlib.patches import FancyArrowPatch
    # Group edges by unordered pair
    undirected_pairs = {}
    for (u,v), w, col in zip(G.edges(), width_scaled, draw_colors):
        key = tuple(sorted([u,v])) if u != v else (u,v)
        undirected_pairs.setdefault(key, []).append((u,v,w,col))
    # Curvature radius base (adjust for number of nodes for readability)
    base_rad = 0.22
    for key, edge_list in undirected_pairs.items():
        for (u,v,w,col) in edge_list:
            x0,y0 = pos[u]; x1,y1 = pos[v]
            if u == v:
                # Self-loop (draw small circular arc above node)
                loop_radius = 0.18
                theta = np.linspace(0.3*math.pi, 0.7*math.pi, 30)
                cx = x0 + loop_radius*np.cos(theta)
                cy = y0 + loop_radius*np.sin(theta)
                ax.plot(cx, cy, color=col, linewidth=max(1.2, w*0.85), alpha=0.9)
                # Arrowhead at end of loop
                end_x = cx[-1]; end_y = cy[-1]
                ah_size = 0.04
                ax.fill([end_x, end_x - ah_size, end_x + ah_size],
                        [end_y, end_y - ah_size*0.6, end_y - ah_size*0.6],
                        color=col, alpha=0.9, linewidth=0)
                continue
            # Curved edge drawing (explicit quadratic Bezier) so that bidirectional edges are clearly separated.
            bidirectional = (len(edge_list) == 2)
            # Vector between nodes
            dx = x1 - x0; dy = y1 - y0
            length = math.hypot(dx, dy) or 1.0
            # Perpendicular unit vector
            perp_x, perp_y = -dy/length, dx/length
            # Curvature magnitude scales with base_rad and distance
            curve_scale = (0.35 if bidirectional else 0.18) * length * base_rad
            # Assign direction (one up, one down) deterministically: for u<v use +, else - (for opposite direction edge sign flips)
            sign = 1 if (u < v) else -1
            # If bidirectional, the second edge in the pair will get opposite sign automatically due to u<v condition flip.
            control_x = (x0 + x1)/2 + perp_x * curve_scale * sign
            control_y = (y0 + y1)/2 + perp_y * curve_scale * sign
            # Sample quadratic Bezier points
            t_vals = np.linspace(0,1,60)
            bx = (1-t_vals)**2 * x0 + 2*(1-t_vals)*t_vals * control_x + t_vals**2 * x1
            by = (1-t_vals)**2 * y0 + 2*(1-t_vals)*t_vals * control_y + t_vals**2 * y1
            ax.plot(bx, by, color=col, linewidth=w, alpha=0.92, solid_capstyle='round')
            # Midpoint (t=0.5) for arrowhead placement (pointing toward v)
            t_mid = 0.5
            mx = (1-t_mid)**2 * x0 + 2*(1-t_mid)*t_mid * control_x + t_mid**2 * x1
            my = (1-t_mid)**2 * y0 + 2*(1-t_mid)*t_mid * control_y + t_mid**2 * y1
            # Tangent derivative at t_mid
            dx_dt = 2*(1-t_mid)*(control_x - x0) + 2*t_mid*(x1 - control_x)
            dy_dt = 2*(1-t_mid)*(control_y - y0) + 2*t_mid*(y1 - control_y)
            mag = math.hypot(dx_dt, dy_dt) or 1.0
            tx, ty = dx_dt/mag, dy_dt/mag
            # Arrowhead triangle at midpoint, pointing along tangent
            ah_len = 0.06 * (1 + 0.4*(w/max(width_scaled)))
            ah_w = ah_len * 0.55
            left_x = mx - tx*ah_len + (-ty)*ah_w
            left_y = my - ty*ah_len + (tx)*ah_w
            right_x = mx - tx*ah_len + (ty)*ah_w
            right_y = my - ty*ah_len + (-tx)*ah_w
            ax.fill([mx, left_x, right_x], [my, left_y, right_y], color=col, alpha=0.95, linewidth=0)
    sizes = [380 + 2400*nodes.set_index('label').loc[n].get('pagerank',0.0) if 'pagerank' in nodes.columns else 400 + 30*G.degree(n, weight='weight') for n in G.nodes()]
    _label_color_map = {
        '긍정': (31/255,119/255,180/255,0.78),
        '부정': (214/255,39/255,40/255,0.78),
        '동정': (255/255,219/255,88/255,0.85),
    }
    node_colors = [_label_color_map.get(n, (0.7,0.7,0.7,0.6)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='#333', linewidths=0.9, node_size=sizes, ax=ax)
    # English label mapping
    en_map = {'긍정':'Positive','부정':'Negative','동정':'Sympathy'}
    nx.draw_networkx_labels(G, pos, labels={n: en_map.get(n, n) for n in G.nodes()},
                            font_size=11, ax=ax, font_family='sans-serif')
    # Edge labels (reuse refined placement + self-loop handling) for sentence-level
    # Edge labels: compute midpoints following the arc geometry approximately (use straight midpoint + perpendicular offset)
    for key, edge_list in undirected_pairs.items():
        if not edge_list:
            continue
        for (u,v,w,col) in edge_list:
            x0,y0 = pos[u]; x1,y1 = pos[v]
            data = G[u][v]
            if u == v:
                ax.text(x0, y0 + 0.28, f"loop {data.get('weight',0)}/{data.get('prob',0):.2f}", fontsize=9,
                        ha='center', va='center', color='black', alpha=0.95,
                        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='#666', alpha=0.55))
                continue
            # Recompute control point (same rule as above) for consistent label offset
            dx = x1 - x0; dy = y1 - y0
            length = math.hypot(dx, dy) or 1.0
            perp_x, perp_y = -dy/length, dx/length
            bidirectional = (len(edge_list) == 2)
            curve_scale = (0.35 if bidirectional else 0.18) * length * base_rad
            sign = 1 if (u < v) else -1
            control_x = (x0 + x1)/2 + perp_x * curve_scale * sign
            control_y = (y0 + y1)/2 + perp_y * curve_scale * sign
            t_mid = 0.5
            mx = (1-t_mid)**2 * x0 + 2*(1-t_mid)*t_mid * control_x + t_mid**2 * x1
            my = (1-t_mid)**2 * y0 + 2*(1-t_mid)*t_mid * control_y + t_mid**2 * y1
            # Label offset slightly further along the perpendicular relative to arc direction
            label_off = 0.05 * sign
            label_x = mx + perp_x * label_off
            label_y = my + perp_y * label_off
            ax.text(label_x, label_y, f"{data.get('weight',0)}/{data.get('prob',0):.2f}", fontsize=9,
                    ha='center', va='center', color='darkslategray', alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#666', alpha=0.55))
    # (Legend removed per new requirement)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Transition Probability')
    # English-only title & caption
    ax.set_title('Sentence  Level Sentiment Transition ')
    ax.text(0.5, -0.065, 'Sentence-level label transitions (edge width=count, color=probability).',
        ha='center', va='top', transform=ax.transAxes, fontsize=9, color='#444')
    fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(out_p/'sentence_transition_network.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    # Matrix (English only)
    mat = edges.pivot_table(index='label', columns='next_label', values='prob', fill_value=0)
    plt.figure(figsize=(4.5,4))
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='Greens')
    # Map axis tick labels to English
    en_map = {'긍정':'Positive','부정':'Negative','동정':'Sympathy'}
    ax_curr = plt.gca()
    ax_curr.set_yticklabels([en_map.get(t.get_text(), t.get_text()) for t in ax_curr.get_yticklabels()], rotation=0)
    ax_curr.set_xticklabels([en_map.get(t.get_text(), t.get_text()) for t in ax_curr.get_xticklabels()], rotation=45, ha='right')
    plt.title('Sentence Transition Probability Matrix')
    plt.xlabel('Next label')
    plt.ylabel('Current label')
    plt.text(0.5, -0.15, 'Row=current label, Column=next label', transform=ax_curr.transAxes,
             ha='center', va='top', fontsize=9, color='#444')
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(out_p/'sentence_transition_matrix.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Sentence transition network plots -> {out_p}')

    # Interactive version (Plotly) sentence-level (English only, arrow at midpoint)
    if _PLOTLY and os.environ.get('INTERACTIVE','0') in {'1','true','True'}:
        try:
            import plotly.graph_objects as go
            inter_dir = out_p / 'interactive'
            inter_dir.mkdir(exist_ok=True)
            # Fixed node positions for designer-style layout
            label_en = {'긍정':'Positive','부정':'Negative','동정':'Sympathy'}
            node_positions = {
                'Negative': (-1, 0),
                'Sympathy': (0, 1.2),
                'Positive': (1, 0)
            }
            # Only plot nodes present in data
            node_list = [n for n in G.nodes() if label_en.get(n,n) in node_positions]
            x_vals = [node_positions[label_en.get(n,n)][0] for n in node_list]
            y_vals = [node_positions[label_en.get(n,n)][1] for n in node_list]
            color_map_hex = {
                'Positive': 'rgba(31,119,180,0.78)',
                'Negative': 'rgba(214,39,40,0.78)',
                'Sympathy': 'rgba(255,219,88,0.85)'
            }
            node_labels_en = [label_en.get(n,n) for n in node_list]
            node_colors_hex = [color_map_hex.get(label_en.get(n,n), 'rgba(180,180,180,0.6)') for n in node_list]
            size_map = {n:s for n,s in zip(G.nodes(), sizes)}
            node_sizes = [size_map[n]*0.06 for n in node_list]
            import math
            edge_traces = []
            label_annotations = []
            total_edges = len(G.edges())
            base_offset = 0.035 + 0.018 * (total_edges/10)**0.5
            for (u,v) in G.edges():
                # Use fixed node positions for both ends
                u_en = label_en.get(u,u)
                v_en = label_en.get(v,v)
                if u_en not in node_positions or v_en not in node_positions:
                    continue
                x0, y0 = node_positions[u_en]
                x1, y1 = node_positions[v_en]
                data = G[u][v]
                if u == v:
                    loop_off = 0.18
                    label_annotations.append(dict(
                        x=x0, y=y0 + loop_off, text=f"loop {data.get('weight',0)}/{data.get('prob',0):.2f}",
                        showarrow=False, font=dict(size=10,color='black'),
                        bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(80,80,80,0.4)', borderwidth=0.5
                    ))
                    continue
                dx = x1 - x0; dy = y1 - y0
                length = math.hypot(dx, dy) or 1.0
                ux, uy = dx/length, dy/length
                # Arrow at midpoint
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                # Probability-based alpha for edge color
                p = float(data.get('prob', 0.0))
                alpha_line = 0.25 + 0.7*max(0.0, min(1.0, p))
                col_line = f'rgba(100,100,100,{alpha_line:.3f})'
                # Draw edge line from source to midpoint
                edge_traces.append(go.Scatter(x=[x0, mid_x], y=[y0, mid_y], mode='lines',
                                              line=dict(color=col_line, width=1.7),
                                              hoverinfo='skip', showlegend=False))
                # Arrowhead at midpoint
                ah_size = 0.07
                left_x = mid_x + (-uy)*ah_size
                left_y = mid_y + (ux)*ah_size
                right_x = mid_x + (uy)*ah_size
                right_y = mid_y + (-ux)*ah_size
                edge_traces.append(go.Scatter(x=[mid_x, left_x, right_x, mid_x], y=[mid_y, left_y, right_y, mid_y],
                                              mode='lines', fill='toself', line=dict(color=col_line, width=1),
                                              hoverinfo='skip', showlegend=False))
                # Label at midpoint, offset perpendicular
                perp_x, perp_y = -uy, ux
                has_reverse = G.has_edge(v,u)
                off = base_offset if (u < v or not has_reverse) else -base_offset
                if has_reverse and (v < u):
                    off = -off
                label_x = mid_x + perp_x * off
                label_y = mid_y + perp_y * off
                label_annotations.append(dict(
                    x=label_x, y=label_y,
                    text=f"{data.get('weight',0)}/{data.get('prob',0):.2f}",
                    showarrow=False,
                    font=dict(size=10,color='darkslategray'),
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='rgba(120,120,120,0.3)', borderwidth=0.5
                ))
            node_trace = go.Scatter(
                x=x_vals, y=y_vals, mode='markers+text',
                text=node_labels_en, textposition='middle center',
                marker=dict(size=node_sizes, color=node_colors_hex, line=dict(color='rgba(50,50,50,0.6)', width=1)),
                hoverinfo='text',
                showlegend=False
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
                margin=dict(l=10,r=140,t=60,b=10),
                paper_bgcolor='white', plot_bgcolor='white'
            )
            fig_i = go.Figure(data=edge_traces + [node_trace], layout=layout)
            fig_i.write_html(inter_dir/'sentence_transition_network_interactive.html')
        except Exception as e:  # pragma: no cover
            print('[WARN] Interactive sentence network failed:', e)


def plot_variability(vari_csv: str = 'results/variability_metrics.csv', out_dir: str = 'results/plots_variability'):
    if not os.path.exists(vari_csv):
        return
    df = pd.read_csv(vari_csv)
    if df.empty:
        return
    out_dir_p = _ensure_dir(out_dir)
    # Scatter: paragraph_std vs sentence_std
    if {'paragraph_std','sentence_std'} <= set(df.columns):
        plt.figure(figsize=(5.2,4.4))
        plt.scatter(df['paragraph_std'], df['sentence_std'], alpha=0.6)
        mx = max(df['paragraph_std'].max(), df['sentence_std'].max())
        plt.plot([0,mx],[0,mx], color='red', linestyle='--', linewidth=1)
        plt.xlabel('문단 표준편차 / Paragraph Std')
        plt.ylabel('문장 표준편차 / Sentence Std')
        plt.title(_title('문단 vs 문장 감정 변동성', 'Paragraph vs Sentence Sentiment Variability'))
        _add_caption('대각선 위: 문장 변동성이 더 큼.', 'Above diagonal: higher within-article sentence variability.')
        plt.tight_layout(rect=[0,0.06,1,1])
        plt.savefig(out_dir_p/'variability_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    if 'ratio_sentence_paragraph' in df.columns:
        plt.figure(figsize=(5.4,4.2))
        sns.histplot(df['ratio_sentence_paragraph'].dropna(), bins=30, kde=True, color='#6A5ACD')
        plt.title(_title('문장/문단 변동성 비율', 'Sentence/Paragraph Variability Ratio'))
        plt.xlabel('비율 (sentence_std / paragraph_std)')
        plt.ylabel('빈도 / Frequency')
        _add_caption('>1: 문장 레벨 변동성이 문단보다 큼.', '>1 means sentence-level variability dominates.')
        plt.tight_layout(rect=[0,0.06,1,1])
        plt.savefig(out_dir_p/'variability_ratio_hist.png', dpi=150, bbox_inches='tight')
        plt.close()
    if 'source' in df.columns and df['source'].notna().any():
        grp = df.groupby('source')['ratio_sentence_paragraph'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=grp, x='source', y='ratio_sentence_paragraph')
        plt.title(_title('매체별 문장/문단 변동성 비율', 'Avg Sentence/Paragraph Variability Ratio by Source'))
        plt.xlabel('매체 / Source')
        plt.ylabel('평균 비율 / Mean Ratio')
        _add_caption('매체별 미시 감정 기복 상대 비교.', 'Relative micro-level tone fluctuation per outlet.')
        plt.tight_layout(rect=[0,0.06,1,1])
        plt.savefig(out_dir_p/'variability_ratio_by_source.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f'[INFO] Variability plots -> {out_dir_p}')


def plot_all(results_dir: str = 'results',
             highlight_top_k: int = 0,
             wordcloud_stopwords: str | None = None,
             wordcloud_mode: str = 'auto',
             wordcloud_pos_filter: bool = False,
             wordcloud_freq_top: int = 0):
    """존재하는 모든 결과 CSV를 탐지해 시각화.

    생성 대상:
      sentiment.csv            -> 기본 플롯 (trend, source box/bar)
      paragraph_flow.csv       -> 문단 곡선/분포
      topic_sentiment_docs.csv + topic_sentiment_summary.csv -> 토픽
      bias_summary.csv         -> bias 지표
    산출물은 results/ 하위 개별 폴더에 저장.
    """
    try:
        index_lines = []
        sent_path = Path(results_dir)/'sentiment.csv'
        if sent_path.exists():
            plot_results(str(sent_path), out_dir=results_dir)
            index_lines += [
                '01_sentiment_trend.png : 월별 평균 감정 점수 추이',
                '01b_sentiment_hist.png : 전체 감정 점수 히스토그램(KDE)',
                '01c_sentiment_hist_by_source.png : 매체별 감정 점수 히스토그램',
                '02_sentiment_box_source.png : 매체별 감정 점수 분포',
                '03_sentiment_bar_source.png : 매체별 평균 감정 점수',
                '04_label_counts.png : 감정 레이블 문서 수 (존재 시)',
                '05_label_mean_prob.png : 평균 레이블 확률 (prob_* 존재 시)',
                '06_label_monthly_count_trend.png : 월별 레이블 문서 수 추이',
                '07_label_monthly_prob_trend.png : 월별 레이블 평균 확률 추이 (prob_* 존재 시)',
                '08_label_monthly_ratio_area.png : 월별 레이블 비율 스택 에어리어',
                '09_score_vs_length_scatter.png : 감정 점수 vs 문서 길이 산점'
            ]
            # Optional Plotly interactive set
            if _PLOTLY and os.environ.get('INTERACTIVE','0') in {'1','true','True'}:
                try:
                    df_sent = pd.read_csv(sent_path)
                    has_date_int = False
                    if 'date' in df_sent.columns:
                        try:
                            raw = (
                                df_sent['date'].astype(str).str.strip()
                                .str.replace('\\.', '-', regex=False)
                                .str.replace('/', '-', regex=False)
                            )
                            raw = raw.str.replace(r'[^0-9\-]', '', regex=True)
                            raw = raw.apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if (len(x)==8 and '-' not in x) else x)
                            df_sent['date_dt'] = pd.to_datetime(raw, errors='coerce', infer_datetime_format=True)
                            if df_sent['date_dt'].notna().any():
                                has_date_int = True
                        except Exception:
                            pass
                    inter_dir = Path(results_dir)/'interactive'
                    inter_dir.mkdir(exist_ok=True)
                    # Monthly trend interactive
                    if has_date_int and 'score' in df_sent.columns:
                        ts = df_sent.groupby(df_sent['date_dt'].dt.to_period('M'))['score'].mean().reset_index()
                        ts['date'] = ts['date_dt'].dt.to_timestamp()
                        fig = px.line(ts, x='date', y='score', markers=True, title='월별 평균 감정 점수 추이 / Monthly Average Sentiment')
                        fig.update_yaxes(title='Score (-1~+1)')
                        fig.write_html(inter_dir/'sentiment_trend.html')
                        index_lines.append('interactive/sentiment_trend.html : 인터랙티브 월별 평균 감정')
                    # Histogram overall
                    if 'score' in df_sent.columns:
                        fig = px.histogram(df_sent, x='score', nbins=30, marginal='box', title='감정 점수 분포 / Sentiment Score Distribution')
                        fig.update_xaxes(title='Score (-1~+1)')
                        fig.write_html(inter_dir/'sentiment_hist.html')
                        index_lines.append('interactive/sentiment_hist.html : 인터랙티브 점수 분포')
                    # Label monthly trend (counts)
                    if has_date_int and 'label' in df_sent.columns:
                        ml = df_sent.groupby([df_sent['date_dt'].dt.to_period('M'), 'label']).size().reset_index(name='count')
                        ml['month'] = ml['date_dt'].dt.to_timestamp()
                        fig = px.line(ml, x='month', y='count', color='label', markers=True, title='월별 레이블 등장 추이 / Monthly Label Count Trend')
                        fig.write_html(inter_dir/'label_monthly_trend.html')
                        index_lines.append('interactive/label_monthly_trend.html : 인터랙티브 레이블 추이')
                    # Topic heatmap interactive
                    t_docs = Path(results_dir)/'topic_sentiment_docs.csv'
                    t_sum = Path(results_dir)/'topic_sentiment_summary.csv'
                    if t_docs.exists() and t_sum.exists():
                        try:
                            docs = pd.read_csv(t_docs)
                            summ = pd.read_csv(t_sum)
                            if 'date' in docs.columns and 'topic' in docs.columns:
                                try:
                                    rawd = (
                                        docs['date'].astype(str).str.strip()
                                        .str.replace('\\.', '-', regex=False)
                                        .str.replace('/', '-', regex=False)
                                    )
                                    rawd = rawd.str.replace(r'[^0-9\-]', '', regex=True)
                                    rawd = rawd.apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if (len(x)==8 and '-' not in x) else x)
                                    docs['date_dt'] = pd.to_datetime(rawd, errors='coerce', infer_datetime_format=True)
                                except Exception:
                                    docs['date_dt'] = pd.NaT
                                docs_f = docs[(docs['topic'].isin(summ['topic'])) & docs['date_dt'].notna()].copy()
                                if not docs_f.empty:
                                    docs_f['month'] = docs_f['date_dt'].dt.to_period('M').dt.to_timestamp()
                                heat = docs_f.pivot_table(index='topic', columns='month', values='score', aggfunc='count', fill_value=0)
                                heat = heat.loc[summ['topic']]
                                if 'Name' in summ.columns:
                                    names = summ.set_index('topic')['Name'].to_dict()
                                    heat.index = [f"{t}:{str(names.get(t,''))[:25]}" for t in heat.index]
                                fig = go.Figure(data=go.Heatmap(z=heat.values, x=[c.strftime('%Y-%m') for c in heat.columns], y=heat.index, colorscale='Blues'))
                                fig.update_layout(title='토픽별 월별 문서 수 히트맵 / Topic-Month Heatmap', xaxis_title='Month', yaxis_title='Topic')
                                fig.write_html(inter_dir/'topic_month_heatmap.html')
                                index_lines.append('interactive/topic_month_heatmap.html : 인터랙티브 토픽-월 히트맵')
                        except Exception:
                            pass
                    # ------------------------------------------------------------------
                    # Interactive network graphs (paragraph & sentence sentiment transitions)
                    # ------------------------------------------------------------------
                    def _interactive_transition_network(edges_csv: Path, nodes_csv: Path, out_name: str, color_scale='Blues'):
                        try:
                            if not (edges_csv.exists() and nodes_csv.exists()):
                                return None
                            import pandas as _pd, networkx as _nx
                            e = _pd.read_csv(edges_csv)
                            n = _pd.read_csv(nodes_csv)
                            if e.empty or n.empty:
                                return None
                            show_network_metrics = True  # always display metrics/text
                            G = _nx.DiGraph()
                            for _, r in e.iterrows():
                                G.add_edge(r['label'], r['next_label'], weight=r['count'], prob=r.get('prob',0.0))
                            # Use spring layout for consistency
                            pos = _nx.spring_layout(G, weight='weight', seed=42)
                            # Build node scatter
                            x_nodes = [pos[k][0] for k in G.nodes()]
                            y_nodes = [pos[k][1] for k in G.nodes()]
                            pr_map = {r['label']: r.get('pagerank',0.0) for _, r in n.iterrows()} if 'pagerank' in n.columns else {}
                            bt_map = {r['label']: r.get('betweenness',0.0) for _, r in n.iterrows()} if 'betweenness' in n.columns else {}
                            cl_map = {r['label']: r.get('clustering',0.0) for _, r in n.iterrows()} if 'clustering' in n.columns else {}
                            sizes = [(12 + 80*pr_map.get(nd,0.0)) for nd in G.nodes()] if pr_map else [(18 + 6*G.degree(nd, weight='weight')) for nd in G.nodes()]
                            if show_network_metrics:
                                node_texts = []
                                for nd in G.nodes():
                                    parts = [nd]
                                    if pr_map: parts.append(f"PR:{pr_map.get(nd,0):.2f}")
                                    if bt_map: parts.append(f"B:{bt_map.get(nd,0):.2f}")
                                    if cl_map: parts.append(f"C:{cl_map.get(nd,0):.2f}")
                                    node_texts.append('<br>'.join(parts))
                                node_trace = go.Scatter(
                                    x=x_nodes, y=y_nodes, mode='markers+text',
                                    text=node_texts, textposition='top center',
                                    hoverinfo='skip',
                                    marker=dict(size=sizes, color='#FFD27F', line=dict(width=1, color='#444'))
                                )
                            else:
                                node_trace = go.Scatter(
                                    x=x_nodes, y=y_nodes, mode='markers+text',
                                    text=list(G.nodes()), textposition='top center',
                                    marker=dict(size=sizes, color='#FFD27F', line=dict(width=1, color='#444'))
                                )
                            # Edge traces (as separate segments)
                            edge_x = []
                            edge_y = []
                            edge_text = []  # for hover fallback
                            edge_label_x = []
                            edge_label_y = []
                            edge_label_text = []
                            probs = []
                            import math
                            for (u,v) in G.edges():
                                x0,y0 = pos[u]; x1,y1 = pos[v]
                                edge_x += [x0,x1,None]
                                edge_y += [y0,y1,None]
                                probs.append(G[u][v].get('prob',0.0))
                                # bidirectional offset calc for label position
                                mid_x = (x0 + x1)/2; mid_y = (y0 + y1)/2
                                dx = x1 - x0; dy = y1 - y0
                                length = math.hypot(dx, dy) or 1.0
                                perp_x, perp_y = -dy/length, dx/length
                                has_reverse = G.has_edge(v,u)
                                offset_scale = 0.04 + 0.02 * (len(G.edges())/10)**0.5
                                off = offset_scale if (u < v or not has_reverse) else -offset_scale
                                if has_reverse and (v < u):
                                    off = -off
                                label_x = mid_x + perp_x * off
                                label_y = mid_y + perp_y * off
                                edge_label_x.append(label_x)
                                edge_label_y.append(label_y)
                                edge_label_text.append(f"{G[u][v]['weight']}/{G[u][v].get('prob',0):.2f}")
                            # Map probabilities to color
                            if probs:
                                import numpy as _np
                                norm_probs = [(p - min(probs)) / ( (max(probs)-min(probs)) or 1) for p in probs]
                                colors = [plt.cm.get_cmap(color_scale)(p) for p in norm_probs]
                                # Flatten RGBA to hex
                                def rgba_to_hex(rgba):
                                    r,g,b,a = [int(c*255) if i<3 else c for i,c in enumerate(rgba)]
                                    return f'rgba({r},{g},{b},{a:.2f})'
                                edge_color_list = [rgba_to_hex(c) for c in colors]
                            else:
                                edge_color_list = 'rgba(50,50,50,0.6)'
                            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                                     line=dict(width=2, color='rgba(80,80,80,0.45)'),
                                                     hoverinfo='skip')
                            # Always-visible edge label layer (as text markers)
                            edge_label_trace = go.Scatter(
                                x=edge_label_x, y=edge_label_y, mode='text',
                                text=edge_label_text,
                                textfont=dict(size=10, color='black'),
                                hoverinfo='skip'
                            )
                            data=[edge_trace, edge_label_trace, node_trace]
                            fig_net = go.Figure(data=data)
                            fig_net.update_layout(title=out_name.replace('.html',''), showlegend=False, margin=dict(l=10,r=10,t=50,b=10))
                            out_path = inter_dir/out_name
                            fig_net.write_html(out_path)
                            return out_path
                        except Exception as _e:
                            print(f'[WARN] interactive network failed ({out_name}): {_e}')
                            return None
                    # Paragraph-level network
                    _interactive_transition_network(Path(results_dir)/'network'/'sentiment_transition_edges.csv',
                                                     Path(results_dir)/'network'/'sentiment_transition_nodes.csv',
                                                     'paragraph_transition_network.html','Blues')
                    # Sentence-level network
                    _interactive_transition_network(Path(results_dir)/'network_sentence'/'sentiment_transition_edges.csv',
                                                     Path(results_dir)/'network_sentence'/'sentiment_transition_nodes.csv',
                                                     'sentence_transition_network.html','Greens')
                except Exception as ie:
                    print(f'[WARN] Interactive plot generation failed: {ie}')
        plot_paragraph_flow(Path(results_dir)/'paragraph_flow.csv', out_dir=str(Path(results_dir)/'plots_paragraph'))
        if Path(results_dir,'plots_paragraph').exists():
            index_lines += [
                'plots_paragraph/00_paragraph_score_violin.png : 전체 문단 감정 점수 분포',
                'plots_paragraph/00_paragraph_score_box.png : 문단 점수 박스/레이블별',
                'plots_paragraph/*_paragraph_flow_article*.png : 기사별 문단 감정 흐름'
            ]
        # Sentence flow (optional)
        plot_sentence_flow(Path(results_dir)/'sentence_flow.csv', out_dir=str(Path(results_dir)/'plots_sentence'))
        if Path(results_dir,'plots_sentence').exists():
            index_lines += [
                'plots_sentence/00_sentence_score_violin.png : 전체 문장 감정 분포',
                'plots_sentence/00_sentence_score_box.png : 문장 점수 박스/레이블별',
                'plots_sentence/00_sentence_length_scatter.png : 문장 길이 vs 감정 점수',
                'plots_sentence/*_sentence_flow_article*.png : 기사별 문장 감정 흐름'
            ]
        plot_topic_results(
            topic_docs_csv=str(Path(results_dir)/'topic_sentiment_docs.csv'),
            topic_summary_csv=str(Path(results_dir)/'topic_sentiment_summary.csv'),
            out_dir=str(Path(results_dir)/'plots_topic')
        )
        if Path(results_dir,'plots_topic').exists():
            index_lines += [
                'plots_topic/topic_docs_count.png : 토픽별 문서 수',
                'plots_topic/topic_avg_score.png : 토픽별 평균 감정 점수',
                'plots_topic/topic_label_composition.png : 토픽별 감정 레이블 비율'
            ]
            if Path(results_dir,'plots_topic','topic_month_heatmap.png').exists():
                index_lines.append('plots_topic/topic_month_heatmap.png : 토픽별 월별 문서 수 히트맵')
        plot_bias(Path(results_dir)/'bias_summary.csv', out_dir=str(Path(results_dir)/'plots_bias'))
        if Path(results_dir,'plots_bias').exists():
            index_lines += [
                'plots_bias/bias_index.png : 매체별 Bias Index',
                'plots_bias/neutral_ratio.png : 매체별 중립 비율',
                'plots_bias/entropy_labels.png : 매체별 레이블 엔트로피',
                'plots_bias/bias_vs_avg.png : Bias vs 평균 감정 (버블=문서 수)'
            ]
        # Sentiment transition network (if edges/nodes present)
        plot_sentiment_transition_network(
            edges_csv=str(Path(results_dir)/'network'/'sentiment_transition_edges.csv'),
            nodes_csv=str(Path(results_dir)/'network'/'sentiment_transition_nodes.csv'),
            out_dir=str(Path(results_dir)/'plots_network'),
            highlight_top_k=highlight_top_k
        )
        if Path(results_dir,'plots_network').exists():
            index_lines += [
                'plots_network/sentiment_transition_network.png : 감정 전이 네트워크',
                'plots_network/sentiment_transition_matrix.png : 감정 전이 확률 행렬'
            ]
            inter_dir = Path(results_dir)/'plots_network'/'interactive'
            if inter_dir.exists():
                if (inter_dir/'sentiment_transition_network_interactive.html').exists():
                    index_lines.append('plots_network/interactive/sentiment_transition_network_interactive.html : 인터랙티브 감정 전이 네트워크')
        # Sentence-level transition network plots
        plot_sentence_transition_network(
            edges_csv=str(Path(results_dir)/'network_sentence'/'sentiment_transition_edges.csv'),
            nodes_csv=str(Path(results_dir)/'network_sentence'/'sentiment_transition_nodes.csv'),
            out_dir=str(Path(results_dir)/'plots_network_sentence'),
            highlight_top_k=highlight_top_k
        )
        if Path(results_dir,'plots_network_sentence').exists():
            index_lines += [
                'plots_network_sentence/sentence_transition_network.png : 문장 감정 전이 네트워크',
                'plots_network_sentence/sentence_transition_matrix.png : 문장 감정 전이 확률 행렬'
            ]
            inter_dir2 = Path(results_dir)/'plots_network_sentence'/'interactive'
            if inter_dir2.exists():
                if (inter_dir2/'sentence_transition_network_interactive.html').exists():
                    index_lines.append('plots_network_sentence/interactive/sentence_transition_network_interactive.html : 인터랙티브 문장 감정 전이 네트워크')
        # Variability plots
        plot_variability('results/variability_metrics.csv', out_dir=str(Path(results_dir)/'plots_variability'))
        if Path(results_dir,'plots_variability').exists():
            index_lines += [
                'plots_variability/variability_scatter.png : 문단 vs 문장 변동성 산점',
                'plots_variability/variability_ratio_hist.png : 문장/문단 변동성 비율 히스토그램',
                'plots_variability/variability_ratio_by_source.png : 매체별 변동성 비율'
            ]
        if index_lines:
            with open(Path(results_dir)/'plots_index.txt','w',encoding='utf-8') as f:
                f.write('\n'.join(index_lines))
            explanations: List[str] = [
                '감정 점수(Sentiment Score): -1 부정 / 0 중립 / +1 긍정 — 기사 어조 polarity.',
                'Bias Index: (긍정-부정)/(긍정+부정) — >0 이면 긍정 기사 우위 (편향 가능성).',
                '레이블 엔트로피(Label Entropy): 감정 다양성 지표 (높을수록 다양).',
                '중립 비율(Neutral Ratio): 중립 레이블 기사 비중.',
                '문단 z-score: 기사 내부 평균 대비 상대 감정 높낮이.',
                '토픽(Topic): 유사 단어/내용을 공유하는 기사 군집.',
                '평균 레이블 확률(Mean Label Probability): 모델이 그 감정을 정답이라 보는 신뢰 평균.',
                '레이블 비율 스택 에어리어: 월별 기사에서 각 감정 레이블이 차지하는 비중.',
                'Word Count: 공백 단위 토큰 수(길이) — 길이와 감정 관계 참고.',
                '감정별 WordCloud: 각 감정 레이블 문서에서 자주 등장하는 단어 시각화.',
                '감정 전이 네트워크(Sentiment Transition Network): 문단 순서에서 감정 레이블 이동 (간선=전이, 두께=횟수, 색=확률).',
                '전이 확률 행렬(Transition Probability Matrix): 현재 레이블 행 기준 다음 레이블 조건부 확률 P(next|current).'
            ]
            with open(Path(results_dir)/'plots_explanation.txt','w',encoding='utf-8') as fe:
                fe.write('\n'.join(explanations))
            print('[INFO] plots_index.txt & plots_explanation.txt 생성')

        # Optional unified dashboard
        if os.environ.get('DASHBOARD','0') in {'1','true','True'}:
            try:
                generate_dashboard(results_dir)
            except Exception as de:
                print(f'[WARN] dashboard generation failed: {de}')
        # Per-label wordclouds (optional) - generate after index so they don't break earlier logic
        try:
            # Advanced stopwords build if needed
            stopwords_path = wordcloud_stopwords
            stopwords_set = None
            if wordcloud_mode != 'none':
                try:
                    from src.preprocess.stopwords_util import load_stopwords
                    freq_corpus = None
                    if wordcloud_freq_top > 0:
                        # use article texts for frequency; pick from processed file if exists
                        proc_file = Path('data/processed/articles_processed.csv')
                        if proc_file.exists():
                            import pandas as _pd
                            df_proc = _pd.read_csv(proc_file)
                            candidate_col = None
                            for c in ['tokenized_sentences','clean_content','content']:
                                if c in df_proc.columns:
                                    candidate_col = c; break
                            if candidate_col:
                                freq_corpus = df_proc[candidate_col].dropna().astype(str).tolist()
                    stopwords_set = load_stopwords(
                        mode=wordcloud_mode,
                        file_path=stopwords_path,
                        use_pos=wordcloud_pos_filter,
                        freq_corpus=freq_corpus,
                        freq_top_n=wordcloud_freq_top
                    )
                except Exception as se:
                    print(f'[WARN] advanced stopwords build 실패: {se}')
            generate_label_wordclouds(
                sentiment_csv=str(Path(results_dir)/'sentiment.csv'),
                out_dir=str(Path(results_dir)/'wordclouds'),
                stopwords_path=stopwords_path if stopwords_set is None else None
            )
            # Note: if stopwords_set built, current generate_label_wordclouds only supports file or inline fallback.
            # Future: extend function to accept direct set.
        except Exception as wce:
            print(f'[WARN] per-label wordcloud failed: {wce}')
    except Exception as e:
        print(f"[WARN] plot_all failed: {e}")


def generate_dashboard(results_dir: str = 'results', out_name: str = 'dashboard.html'):
    """Generate a single bilingual HTML dashboard embedding images and (if present) interactive HTML via iframe.

    The dashboard auto-discovers key PNG plots from results root and subfolders plus interactive/*.html.
    """
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(results_dir)
    # Collect images (sorted for stable order)
    img_patterns = [
        '01_sentiment_trend.png','01b_sentiment_hist.png','01c_sentiment_hist_by_source.png',
        '02_sentiment_box_source.png','03_sentiment_bar_source.png','04_label_counts.png',
        '05_label_mean_prob.png','06_label_monthly_count_trend.png','07_label_monthly_prob_trend.png',
        '08_label_monthly_ratio_area.png','09_score_vs_length_scatter.png'
    ]
    # Paragraph/topic/bias subfolders
    paragraph_imgs = sorted([p for p in (root/'plots_paragraph').glob('*.png')]) if (root/'plots_paragraph').exists() else []
    topic_imgs = sorted([p for p in (root/'plots_topic').glob('*.png')]) if (root/'plots_topic').exists() else []
    bias_imgs = sorted([p for p in (root/'plots_bias').glob('*.png')]) if (root/'plots_bias').exists() else []
    core_imgs = [root/p for p in img_patterns if (root/p).exists()]
    # Interactive htmls
    interactive_dir = root/'interactive'
    interactive_files = sorted(interactive_dir.glob('*.html')) if interactive_dir.exists() else []

    # Load explanations if available
    explanations_text = ''
    expl_file = root/'plots_explanation.txt'
    if expl_file.exists():
        explanations_text = expl_file.read_text(encoding='utf-8')

    # HTML building
    def img_section(title_ko, title_en, paths):
        if not paths: return ''
        rows = []
        for p in paths:
            rel = p.relative_to(root)
            rows.append(f"<figure><img src='{rel.as_posix()}' loading='lazy'><figcaption>{rel.as_posix()}</figcaption></figure>")
        return f"<section><h2>{title_ko}<br><small>{title_en}</small></h2>{''.join(rows)}</section>"

    def iframe_section(title_ko, title_en, paths):
        if not paths: return ''
        rows = []
        for p in paths:
            rel = p.relative_to(root)
            rows.append(f"<div class='iframe-wrapper'><h4>{rel.name}</h4><iframe src='{rel.as_posix()}' loading='lazy'></iframe></div>")
        return f"<section><h2>{title_ko}<br><small>{title_en}</small></h2>{''.join(rows)}</section>"

    html = f"""<!DOCTYPE html>
<html lang='ko'>
<head>
    <meta charset='utf-8'/>
    <title>감정/토픽 분석 대시보드 | Sentiment & Topic Dashboard</title>
    <style>
        body {{ font-family: -apple-system,BlinkMacSystemFont,'AppleGothic','NanumGothic','Segoe UI',sans-serif; margin:0; padding:1.2rem 2rem; background:#fafafa; }}
        header {{ margin-bottom:1.5rem; }}
        h1 {{ line-height:1.2; }}
        section {{ margin-bottom:2.5rem; }}
        figure {{ display:inline-block; margin:0 0.8rem 1rem 0; background:#fff; padding:0.5rem; border:1px solid #ddd; box-shadow:0 1px 2px rgba(0,0,0,.05); max-width:340px; }}
        figure img {{ max-width:320px; height:auto; display:block; }}
        figcaption {{ font-size:0.75rem; color:#555; margin-top:0.3rem; word-break:break-all; }}
        .iframe-wrapper {{ margin:1rem 0; background:#fff; padding:0.5rem 0.8rem 1rem; border:1px solid #ddd; box-shadow:0 1px 2px rgba(0,0,0,.05); }}
        iframe {{ width:100%; height:480px; border:none; background:#fff; }}
        .explanations {{ white-space:pre-wrap; font-size:0.8rem; background:#fff; padding:1rem; border:1px solid #ddd; }}
        nav a {{ margin-right:1rem; font-size:0.85rem; }}
        .grid-note {{ font-size:0.75rem; color:#666; }}
    </style>
</head>
<body>
    <header>
        <h1>감정 & 토픽 분석 대시보드<br><small>Sentiment & Topic Analysis Dashboard</small></h1>
        <nav>
            <a href='#sentiment'>Sentiment</a>
            <a href='#paragraph'>Paragraph</a>
            <a href='#topic'>Topic</a>
            <a href='#bias'>Bias</a>
            <a href='#interactive'>Interactive</a>
            <a href='#explanations'>Glossary</a>
        </nav>
        <p class='grid-note'>이미지들은 정적 PNG, Interactive 섹션은 Plotly HTML iframe 입니다. / Static PNG images; interactive Plotly iframes below.</p>
    </header>
    <main>
        <a id='sentiment'></a>
        {img_section('감정 (전역)', 'Sentiment (Global)', core_imgs)}
        <a id='paragraph'></a>
        {img_section('문단 흐름', 'Paragraph Flow', paragraph_imgs)}
        <a id='topic'></a>
        {img_section('토픽 통계', 'Topic Statistics', topic_imgs)}
        <a id='bias'></a>
        {img_section('편향 지표', 'Bias Metrics', bias_imgs)}
        <a id='interactive'></a>
        {iframe_section('인터랙티브 플롯', 'Interactive Plots', interactive_files)}
        <a id='explanations'></a>
        <section>
            <h2>용어 설명 / Glossary</h2>
            <div class='explanations'>{explanations_text}</div>
        </section>
    </main>
</body>
</html>"""
    out_file = root/out_name
    out_file.write_text(html, encoding='utf-8')
    print(f"[INFO] Dashboard generated -> {out_file}")