#!/usr/bin/env python3
"""
Language Detection Script for Mapillary Traffic Sign Text Data

This script adds language detection to det_with_text.csv using fastText's
pre-trained language identification model (lid.176.bin).

Requirements:
    - fasttext library
    - lid.176.bin model (automatically downloaded if not present)

Output:
    - det_with_text_fasttext.csv with added language_fasttext and confidence_fasttext columns
    - Statistics on detected languages

Author: Created for GeoguessrAI project
Date: 2026-02-09
"""

import os
import sys
import json
import pandas as pd
import subprocess
from pathlib import Path
from collections import Counter
import urllib.request


def install_fasttext():
    """Install fasttext library if not already installed."""
    try:
        import fasttext
        print("✓ fasttext library is already installed")
        return True
    except ImportError:
        print("Installing fasttext library...")
        try:
            # Try fasttext-wheel first for better compatibility
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext-wheel"])
            print("✓ fasttext library installed successfully")
            return True
        except subprocess.CalledProcessError:
            # Fallback to regular fasttext
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
                print("✓ fasttext library installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install fasttext: {e}")
                return False


def download_fasttext_model(model_path):
    """
    Download the fastText language identification model if not present.

    Args:
        model_path (str): Path where the model should be saved

    Returns:
        bool: True if model is available, False otherwise
    """
    if os.path.exists(model_path):
        print(f"✓ Model already exists at {model_path}")
        return True

    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    print(f"Downloading fastText model from {model_url}...")
    print("This may take a few minutes (file size: ~131 MB)...")

    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✓ Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False


def combine_texts(texts_str):
    """
    Combine all text strings from the 'texts' column.

    Args:
        texts_str (str): JSON string containing list of text strings

    Returns:
        str: Combined text or empty string if parsing fails
    """
    if pd.isna(texts_str) or texts_str == '[]':
        return ""

    try:
        texts_list = json.loads(texts_str)
        # Filter out empty strings and join with space
        combined = " ".join([text.strip() for text in texts_list if text.strip()])
        return combined
    except (json.JSONDecodeError, TypeError):
        return ""


def detect_language(text, model):
    """
    Detect language using fastText model.

    Args:
        text (str): Text to analyze
        model: fastText model object

    Returns:
        tuple: (language_code, confidence) or ('unknown', 0.0) if text is empty/short
    """
    # Handle empty or very short text
    if not text or len(text.strip()) < 2:
        return 'unknown', 0.0

    # Clean text: fastText expects single line
    text_clean = text.replace('\n', ' ').replace('\r', ' ').strip()

    try:
        # Predict language (returns top prediction)
        predictions = model.predict(text_clean, k=1)

        # Extract language code (remove '__label__' prefix)
        language = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])

        return language, confidence
    except Exception as e:
        print(f"Warning: Error detecting language for text '{text[:50]}...': {e}")
        return 'error', 0.0


def process_csv(input_path, output_path, model_path):
    """
    Main processing function to add language detection to CSV.

    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to output CSV file
        model_path (str): Path to fastText model

    Returns:
        pd.DataFrame: Processed dataframe with language columns
    """
    # Load fastText model
    print(f"\nLoading fastText model from {model_path}...")
    try:
        import fasttext
        # Suppress fastText warnings
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

    # Read CSV
    print(f"\nReading CSV from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to read CSV: {e}")
        return None

    # Initialize new columns
    df['language_fasttext'] = 'no_text'
    df['confidence_fasttext'] = 0.0

    # Process rows where n_regions > 0
    print("\nProcessing rows with detected text...")
    rows_with_text = df[df['n_regions'] > 0].index
    print(f"Found {len(rows_with_text)} rows with n_regions > 0")

    processed = 0
    for idx in rows_with_text:
        # Combine all texts
        combined_text = combine_texts(df.at[idx, 'texts'])

        if combined_text:
            # Detect language
            language, confidence = detect_language(combined_text, model)
            df.at[idx, 'language_fasttext'] = language
            df.at[idx, 'confidence_fasttext'] = confidence
            processed += 1
        else:
            df.at[idx, 'language_fasttext'] = 'empty'
            df.at[idx, 'confidence_fasttext'] = 0.0

    print(f"✓ Processed {processed} rows with valid text")

    # Save output
    print(f"\nSaving results to {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        print(f"✓ Saved successfully")
    except Exception as e:
        print(f"✗ Failed to save CSV: {e}")
        return None

    return df


def show_statistics(df):
    """
    Display statistics about detected languages.

    Args:
        df (pd.DataFrame): Processed dataframe
    """
    print("\n" + "="*70)
    print("LANGUAGE DETECTION STATISTICS")
    print("="*70)

    # Overall statistics
    total_rows = len(df)
    rows_with_regions = (df['n_regions'] > 0).sum()
    rows_with_language = df[~df['language_fasttext'].isin(['no_text', 'empty', 'unknown'])].shape[0]

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Rows with detected regions (n_regions > 0): {rows_with_regions:,}")
    print(f"Rows with language detected: {rows_with_language:,}")

    # Language distribution
    print("\n" + "-"*70)
    print("LANGUAGE DISTRIBUTION (Top 20)")
    print("-"*70)

    # Filter out special categories
    language_counts = df[~df['language_fasttext'].isin(['no_text', 'empty'])]['language_fasttext'].value_counts()

    print(f"\n{'Language':<15} {'Count':>10} {'Percentage':>12} {'Avg Confidence':>15}")
    print("-"*70)

    for lang, count in language_counts.head(20).items():
        percentage = (count / rows_with_regions) * 100
        avg_conf = df[df['language_fasttext'] == lang]['confidence_fasttext'].mean()
        print(f"{lang:<15} {count:>10,} {percentage:>11.2f}% {avg_conf:>14.4f}")

    # Confidence statistics
    print("\n" + "-"*70)
    print("CONFIDENCE STATISTICS")
    print("-"*70)

    # Only for rows with actual language detection
    conf_data = df[~df['language_fasttext'].isin(['no_text', 'empty', 'unknown'])]['confidence_fasttext']

    if len(conf_data) > 0:
        print(f"\nMean confidence: {conf_data.mean():.4f}")
        print(f"Median confidence: {conf_data.median():.4f}")
        print(f"Min confidence: {conf_data.min():.4f}")
        print(f"Max confidence: {conf_data.max():.4f}")
        print(f"Std deviation: {conf_data.std():.4f}")

        # Confidence ranges
        print("\nConfidence ranges:")
        ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in ranges:
            count = ((conf_data >= low) & (conf_data < high)).sum()
            pct = (count / len(conf_data)) * 100
            print(f"  {low:.1f} - {high:.1f}: {count:>6,} ({pct:>5.2f}%)")

    # Sample results
    print("\n" + "-"*70)
    print("SAMPLE RESULTS (First 10 with detected language)")
    print("-"*70)

    samples = df[~df['language_fasttext'].isin(['no_text', 'empty', 'unknown'])].head(10)

    for idx, row in samples.iterrows():
        combined_text = combine_texts(row['texts'])
        print(f"\nRow {idx}:")
        print(f"  Text: {combined_text[:60]}{'...' if len(combined_text) > 60 else ''}")
        print(f"  Language: {row['language_fasttext']}")
        print(f"  Confidence: {row['confidence_fasttext']:.4f}")
        print(f"  Image: {os.path.basename(row['image_path'])}")


def main():
    """Main execution function."""
    print("="*70)
    print("FASTTEXT LANGUAGE DETECTION FOR TRAFFIC SIGN TEXT")
    print("="*70)

    # Set up paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "det_with_text.csv"
    output_csv = script_dir / "det_with_text_fasttext.csv"
    model_path = script_dir / "lid.176.bin"

    print(f"\nInput CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Model path: {model_path}")

    # Check if input file exists
    if not input_csv.exists():
        print(f"\n✗ Error: Input file not found at {input_csv}")
        return 1

    # Step 1: Install fasttext
    print("\n" + "-"*70)
    print("STEP 1: Installing fasttext library")
    print("-"*70)
    if not install_fasttext():
        return 1

    # Step 2: Download model
    print("\n" + "-"*70)
    print("STEP 2: Downloading fastText model")
    print("-"*70)
    if not download_fasttext_model(str(model_path)):
        return 1

    # Step 3: Process CSV
    print("\n" + "-"*70)
    print("STEP 3: Processing CSV file")
    print("-"*70)
    df = process_csv(str(input_csv), str(output_csv), str(model_path))

    if df is None:
        return 1

    # Step 4: Show statistics
    print("\n" + "-"*70)
    print("STEP 4: Analyzing results")
    print("-"*70)
    show_statistics(df)

    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nOutput saved to: {output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())