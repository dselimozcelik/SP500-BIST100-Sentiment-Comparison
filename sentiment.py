#!/usr/bin/env python3
"""
Step 4: Sentiment Scoring
Use FinBERT to score sentiment of weekend news articles.
Aggregate sentiment by weekend for regression analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = Path("data/weekend_news.parquet")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# FinBERT model
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32  # Process in batches for efficiency


def load_finbert_model():
    """Load FinBERT sentiment analysis model"""
    logger.info(f"Loading FinBERT model: {MODEL_NAME}")

    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    logger.info(f"Using device: {device_name}")

    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        max_length=512,
        truncation=True
    )

    logger.info("✓ Model loaded successfully")
    return sentiment_pipeline


def score_sentiment(texts, sentiment_pipeline):
    """
    Score sentiment for a list of texts.
    Returns: DataFrame with positive, negative, neutral scores
    """
    logger.info(f"Scoring sentiment for {len(texts)} articles...")

    results = []

    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        # Get predictions
        predictions = sentiment_pipeline(batch)

        # Convert to structured format
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']

            # Initialize scores
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            scores[label] = score

            results.append(scores)

        if (i + BATCH_SIZE) % 1000 == 0:
            logger.info(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} articles...")

    return pd.DataFrame(results)


def aggregate_weekend_sentiment(df):
    """Aggregate sentiment scores by weekend"""
    logger.info("Aggregating sentiment by weekend...")

    # Group by weekend_date
    weekend_groups = df.groupby('weekend_date')

    aggregated = weekend_groups.agg({
        'sentiment_positive': ['mean', 'std', 'min', 'max'],
        'sentiment_negative': ['mean', 'std', 'min', 'max'],
        'sentiment_neutral': ['mean', 'std', 'min', 'max'],
        'text': 'count'  # Article count
    }).reset_index()

    # Flatten column names
    aggregated.columns = [
        'weekend_date',
        'sentiment_positive_mean', 'sentiment_positive_std', 'sentiment_positive_min', 'sentiment_positive_max',
        'sentiment_negative_mean', 'sentiment_negative_std', 'sentiment_negative_min', 'sentiment_negative_max',
        'sentiment_neutral_mean', 'sentiment_neutral_std', 'sentiment_neutral_min', 'sentiment_neutral_max',
        'article_count'
    ]

    # Calculate composite sentiment score (positive - negative)
    aggregated['sentiment_composite'] = aggregated['sentiment_positive_mean'] - aggregated['sentiment_negative_mean']

    logger.info(f"✓ Aggregated sentiment for {len(aggregated)} weekends")

    return aggregated


def main():
    """Main sentiment scoring pipeline"""
    logger.info("="*60)
    logger.info("STEP 4: SENTIMENT SCORING")
    logger.info("="*60)

    # Load weekend news
    logger.info(f"\nLoading data from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please run weekend_filter.py first")
        return None

    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"Loaded {len(df)} weekend articles")
    logger.info(f"Covering {df['weekend_date'].nunique()} weekends")

    # Load FinBERT model
    sentiment_pipeline = load_finbert_model()

    # Score sentiment for all articles
    sentiment_scores = score_sentiment(df['text'].tolist(), sentiment_pipeline)

    # Add sentiment scores to dataframe
    df['sentiment_positive'] = sentiment_scores['positive']
    df['sentiment_negative'] = sentiment_scores['negative']
    df['sentiment_neutral'] = sentiment_scores['neutral']

    # Save detailed results with individual article scores
    detailed_output = OUTPUT_DIR / "weekend_news_with_sentiment.parquet"
    df.to_parquet(detailed_output, index=False)
    logger.info(f"\n✓ Saved detailed sentiment scores to {detailed_output}")

    # Aggregate by weekend
    weekend_sentiment = aggregate_weekend_sentiment(df)

    # Save aggregated results
    aggregated_output = OUTPUT_DIR / "weekend_sentiment.parquet"
    weekend_sentiment.to_parquet(aggregated_output, index=False)
    logger.info(f"✓ Saved aggregated weekend sentiment to {aggregated_output}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SENTIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total weekends: {len(weekend_sentiment)}")
    logger.info(f"Average articles per weekend: {weekend_sentiment['article_count'].mean():.1f}")
    logger.info(f"\nSentiment Statistics (mean across all articles):")
    logger.info(f"  Positive: {df['sentiment_positive'].mean():.3f} ± {df['sentiment_positive'].std():.3f}")
    logger.info(f"  Negative: {df['sentiment_negative'].mean():.3f} ± {df['sentiment_negative'].std():.3f}")
    logger.info(f"  Neutral:  {df['sentiment_neutral'].mean():.3f} ± {df['sentiment_neutral'].std():.3f}")
    logger.info(f"\nComposite Sentiment (weekend-level):")
    logger.info(f"  Mean: {weekend_sentiment['sentiment_composite'].mean():.3f}")
    logger.info(f"  Std:  {weekend_sentiment['sentiment_composite'].std():.3f}")
    logger.info(f"  Min:  {weekend_sentiment['sentiment_composite'].min():.3f}")
    logger.info(f"  Max:  {weekend_sentiment['sentiment_composite'].max():.3f}")

    # Show sample weekends
    logger.info("\n" + "="*60)
    logger.info("SAMPLE WEEKEND SENTIMENT")
    logger.info("="*60)
    for _, row in weekend_sentiment.head(5).iterrows():
        logger.info(f"\nWeekend: {row['weekend_date']}")
        logger.info(f"  Articles: {row['article_count']}")
        logger.info(f"  Positive: {row['sentiment_positive_mean']:.3f} (±{row['sentiment_positive_std']:.3f})")
        logger.info(f"  Negative: {row['sentiment_negative_mean']:.3f} (±{row['sentiment_negative_std']:.3f})")
        logger.info(f"  Composite: {row['sentiment_composite']:.3f}")

    return weekend_sentiment


if __name__ == "__main__":
    df = main()
