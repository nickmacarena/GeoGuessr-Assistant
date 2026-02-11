#!/usr/bin/env python3
"""
Compare fastText and lingua-py language detection results side-by-side.
"""

import pandas as pd
import json
from collections import defaultdict

def load_data():
    """Load both result CSVs."""
    print("Loading data...")
    df_fasttext = pd.read_csv('det_with_text_fasttext.csv')
    df_lingua = pd.read_csv('det_with_text_lingua.csv')

    # Merge on index (should be same order)
    df = df_fasttext.copy()
    df['language_lingua'] = df_lingua['language_lingua']
    df['confidence_lingua'] = df_lingua['confidence_lingua']

    # Filter to rows with text detected
    df = df[df['n_regions'] > 0].copy()

    return df

def parse_texts(texts_str):
    """Parse JSON texts array."""
    try:
        texts = json.loads(texts_str)
        return ' '.join(texts) if texts else ''
    except:
        return ''

def analyze_agreement(df):
    """Analyze where models agree/disagree."""
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)

    # Get rows with predictions from both
    df_both = df[
        (df['language_fasttext'].notna()) &
        (df['language_fasttext'] != 'unknown') &
        (df['language_lingua'].notna()) &
        (df['language_lingua'] != 'UNKNOWN')
    ].copy()

    # Normalize language codes (fasttext uses lowercase)
    df_both['lang_fast_norm'] = df_both['language_fasttext'].str.lower()
    df_both['lang_ling_norm'] = df_both['language_lingua'].str.lower()

    # Agreement
    df_both['agree'] = df_both['lang_fast_norm'] == df_both['lang_ling_norm']

    total = len(df_both)
    agree = df_both['agree'].sum()
    disagree = total - agree

    print(f"\nTotal predictions from both models: {total:,}")
    print(f"Models AGREE: {agree:,} ({agree/total*100:.1f}%)")
    print(f"Models DISAGREE: {disagree:,} ({disagree/total*100:.1f}%)")

    return df_both

def show_examples(df_both, category, n=20):
    """Show examples of a specific category."""
    print("\n" + "="*80)
    print(category.upper())
    print("="*80)

    if category == "high agreement":
        # Both models agree and both have high confidence
        subset = df_both[
            (df_both['agree'] == True) &
            (df_both['confidence_fasttext'] > 0.7) &
            (df_both['confidence_lingua'] > 0.7)
        ].sort_values('confidence_fasttext', ascending=False).head(n)

    elif category == "disagreement":
        # Models disagree
        subset = df_both[df_both['agree'] == False].head(n)

    elif category == "fasttext confident":
        # fastText confident but lingua uncertain or disagrees
        subset = df_both[
            (df_both['confidence_fasttext'] > 0.8) &
            (df_both['confidence_lingua'] < 0.5)
        ].sort_values('confidence_fasttext', ascending=False).head(n)

    elif category == "lingua confident":
        # lingua confident but fastText uncertain or disagrees
        subset = df_both[
            (df_both['confidence_lingua'] > 0.8) &
            (df_both['confidence_fasttext'] < 0.5)
        ].sort_values('confidence_lingua', ascending=False).head(n)

    else:
        return

    print(f"\nShowing {len(subset)} examples:\n")

    for idx, row in subset.iterrows():
        text = parse_texts(row['texts'])
        fast_lang = row['language_fasttext']
        fast_conf = row['confidence_fasttext']
        ling_lang = row['language_lingua']
        ling_conf = row['confidence_lingua']

        # Truncate long text
        text_display = text[:60] + "..." if len(text) > 60 else text

        print(f"Text: '{text_display}'")
        print(f"  fastText: {fast_lang:6s} (conf: {fast_conf:.3f})")
        print(f"  lingua:   {ling_lang:6s} (conf: {ling_conf:.3f})")

        if row['agree']:
            print(f"  ✅ AGREE")
        else:
            print(f"  ❌ DISAGREE")
        print()

def show_disagreement_by_language(df_both):
    """Show which languages have most disagreements."""
    print("\n" + "="*80)
    print("DISAGREEMENTS BY LANGUAGE PAIR")
    print("="*80)

    disagreements = df_both[df_both['agree'] == False]

    # Count disagreement pairs
    pairs = defaultdict(int)
    for _, row in disagreements.iterrows():
        pair = f"{row['language_fasttext']} vs {row['language_lingua']}"
        pairs[pair] += 1

    # Sort by frequency
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:15]

    print("\nTop 15 disagreement patterns:\n")
    for pair, count in sorted_pairs:
        print(f"{count:5,} times: {pair}")

def show_language_specific_examples(df_both, n_per_lang=5):
    """Show examples for common languages."""
    print("\n" + "="*80)
    print("LANGUAGE-SPECIFIC EXAMPLES")
    print("="*80)

    # Common languages to check
    languages = ['en', 'zh', 'ja', 'fr', 'de', 'es', 'ko']

    for lang in languages:
        # Get examples where both models agree on this language
        subset = df_both[
            (df_both['lang_fast_norm'] == lang) &
            (df_both['lang_ling_norm'] == lang)
        ].head(n_per_lang)

        if len(subset) == 0:
            continue

        print(f"\n{lang.upper()} - Both models agree ({len(subset)} examples):")
        for _, row in subset.iterrows():
            text = parse_texts(row['texts'])
            text_display = text[:50] + "..." if len(text) > 50 else text
            print(f"  '{text_display}' (fast:{row['confidence_fasttext']:.2f}, ling:{row['confidence_lingua']:.2f})")

def main():
    print("Loading and comparing fastText vs lingua-py results...")

    df = load_data()
    print(f"Loaded {len(df):,} rows with text detected")

    # Analyze agreement
    df_both = analyze_agreement(df)

    # Show different categories of examples
    show_examples(df_both, "high agreement", n=15)
    show_examples(df_both, "disagreement", n=20)
    show_examples(df_both, "fasttext confident", n=15)
    show_examples(df_both, "lingua confident", n=15)

    # Show disagreement patterns
    show_disagreement_by_language(df_both)

    # Show language-specific examples
    show_language_specific_examples(df_both, n_per_lang=8)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()