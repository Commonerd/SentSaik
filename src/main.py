"""main.py
Extended full pipeline execution script.
"""
from __future__ import annotations
import argparse
import os

from src.preprocess.clean_text import preprocess_articles
from src.analysis.sentiment_model import analyze_sentiment
from src.analysis.paragraph_flow import build_paragraph_flow
from src.analysis.sentence_flow import build_sentence_flow
from src.visualize.plot_sentiment import plot_results, plot_all
from src.analysis.sentiment_transition_network import build_sentiment_transition_network


def parse_args():
    p = argparse.ArgumentParser()

    # Basic
    p.add_argument('--keywords', nargs='+', default=['홍사익', '전범'])
    p.add_argument('--sources', nargs='+', default=['경향신문', '조선일보', '동아일보'])
    p.add_argument('--skip-collect', action='store_true', help='Skip collection step if data/raw already exists')

    # Sentiment analysis
    p.add_argument('--method', choices=['lexicon', 'transformer', 'transformer-multi', 'zero-shot'], default='lexicon')
    p.add_argument('--model', default='skt/kobert-base-v1', help='(transformer) Model name')
    p.add_argument('--lexicon', default='models/sentiment_lexicon_ko.tsv')
    p.add_argument('--zero-shot-labels', type=str, help='Comma-separated labels (zero-shot mode)')
    p.add_argument('--zero-shot-model', default='joeddav/xlm-roberta-large-xnli')
    p.add_argument('--limit', type=int, help='Limit number of documents for sentiment analysis')
    p.add_argument('--three-labels', action='store_true', help='For zero-shot, fix labels to 긍정,부정,동정 (ignores --zero-shot-labels)')

    # Extended features
    p.add_argument('--paragraph-flow', action='store_true')
    p.add_argument('--paragraph-method', choices=['lexicon','transformer','transformer-multi','zero-shot'], default=None, help='Paragraph sentiment method (default: same as global method)')
    p.add_argument('--sentence-flow', action='store_true', help='Generate sentence-level sentiment curve')
    p.add_argument('--sentence-method', choices=['lexicon','transformer','transformer-multi','zero-shot'], default=None, help='Sentence sentiment method (default: same as global method)')
    p.add_argument('--sentence-limit-articles', type=int, help='Limit number of articles for sentence sentiment (performance)')
    p.add_argument('--sentence-max-per-article', type=int, help='Limit max number of sentences per article')
    p.add_argument('--sentence-splitter', choices=['regex','kss'], default='regex', help='Sentence splitting method (use kss if installed)')
    p.add_argument('--sentence-transition', action='store_true', help='Generate sentence-level sentiment transition network')
    p.add_argument('--variability-metrics', action='store_true', help='Compute paragraph vs sentence sentiment variability metrics')
    p.add_argument('--topic', action='store_true')
    p.add_argument('--bias-metrics', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--all-plots', action='store_true', help='Generate all possible plots including paragraph/topic/bias')

    # Label reduction (5->3) option
    p.add_argument('--reduce-labels', action='store_true', help='Collapse sentiment labels to 3 (positive/negative/sympathy) for all subsequent analysis/visualization')
    p.add_argument('--label-map', default='config/label_map_3.json', help='Label mapping JSON path (original->reduced)')

    return p.parse_args()


def main():
    args = parse_args()

    # zero-shot 3-label shortcut
    if args.method == 'zero-shot' and args.three_labels:
        args.zero_shot_labels = '긍정,부정,동정'
        if args.reduce_labels:
            print('[INFO] --three-labels 사용: 이미 3라벨이므로 --reduce-labels 옵션은 무시됩니다.')
            args.reduce_labels = False

    # 1. Preprocessing
    preprocess_articles('data/raw', 'data/processed')

    # 2. Sentiment analysis
    sentiment_csv = analyze_sentiment(
        'data/processed',
        'results/sentiment.csv',
        method=args.method,
        lexicon_path=args.lexicon,
        transformer_model=args.model,
        zero_shot_labels=args.zero_shot_labels,
        zero_shot_model=args.zero_shot_model,
        limit=args.limit,
    )

    # (Optional) Apply label reduction: sentiment.csv overwrite
    if args.reduce_labels:
        try:
            from src.utils.label_reduction import reduce_sentiment_file
            sentiment_csv = reduce_sentiment_file(sentiment_csv, args.label_map)
        except Exception as lr_err:
            print(f"[WARN] Label reduction (sentiment) failed: {lr_err}")

    # 3. Paragraph sentiment curve
    if args.paragraph_flow:
        para_method = args.paragraph_method or args.method
        build_paragraph_flow(
            method=para_method,
            zero_shot_labels=args.zero_shot_labels,
            zero_shot_model=args.zero_shot_model,
            transformer_model=args.model,
        )
        if args.reduce_labels:
            # Reduce labels in paragraph_flow.csv
            try:
                from src.utils.label_reduction import reduce_flow_file
                reduce_flow_file('results/paragraph_flow.csv', args.label_map)
            except Exception as e:
                print(f"[WARN] Paragraph flow label reduction failed: {e}")
        # Build sentiment transition network immediately after paragraph flow
        try:
            build_sentiment_transition_network(min_count=args.transition_min_count)
        except Exception as e:
            print(f"[WARN] Sentiment transition network build failed: {e}")
        if args.transition_monthly:
            try:
                from src.analysis.sentiment_transition_network import build_monthly_transition_networks
                build_monthly_transition_networks(
                    min_count=args.transition_min_count,
                    highlight_top_k=args.transition_top_k,
                    max_months=args.transition_monthly_max
                )
            except Exception as me:
                print(f"[WARN] Monthly transition networks failed: {me}")

    # 3b. Sentence sentiment curve
    if args.sentence_flow:
        sent_method = args.sentence_method or args.method
        try:
            build_sentence_flow(
                method=sent_method,
                lexicon_path=args.lexicon,
                transformer_model=args.model,
                zero_shot_labels=args.zero_shot_labels,
                zero_shot_model=args.zero_shot_model,
                limit_articles=args.sentence_limit_articles,
                max_sentences_per_article=args.sentence_max_per_article,
                sentence_splitter=args.sentence_splitter,
            )
        except Exception as se:
            print(f"[WARN] Sentence flow failed: {se}")
        if args.reduce_labels:
            try:
                from src.utils.label_reduction import reduce_flow_file
                reduce_flow_file('results/sentence_flow.csv', args.label_map)
            except Exception as e:
                print(f"[WARN] Sentence flow label reduction failed: {e}")
        # Sentence-level transition network (optional)
        if args.sentence_transition:
            try:
                from src.analysis.sentence_transition_network import build_sentence_transition_network
                build_sentence_transition_network(min_count=args.transition_min_count)
            except Exception as ste:
                print(f"[WARN] Sentence transition network failed: {ste}")

    # 4. Visualization
    if not args.no_plots:
        if args.all_plots:
            plot_all('results',
                     highlight_top_k=args.transition_top_k,
                     wordcloud_stopwords=args.wordcloud_stopwords,
                     wordcloud_mode=args.wordcloud_stopwords_mode,
                     wordcloud_pos_filter=args.wordcloud_pos_filter,
                     wordcloud_freq_top=args.wordcloud_freq_top)
        else:
            plot_results(sentiment_csv)

    print('[DONE] Pipeline completed.')


if __name__ == '__main__':
    main()
