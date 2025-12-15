#!/usr/bin/env python3
"""
Step 3: Weekend Window Filter
Filter news articles that fall within weekend windows (Friday 21:00 UTC - Sunday 23:59 UTC).
Group articles by weekend_start_date and prepare for sentiment analysis.
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = Path("data/combined_news.parquet")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Weekend definition: Friday 21:00 UTC through Sunday 23:59 UTC
WEEKEND_START_DAY = 4  # Friday (0=Monday)
WEEKEND_START_HOUR = 21  # 21:00 UTC
WEEKEND_END_DAY = 6  # Sunday
WEEKEND_END_HOUR = 23
WEEKEND_END_MINUTE = 59


def is_weekend_time(dt):
    """
    Check if a datetime falls within the weekend window.
    Weekend: Friday 21:00 UTC - Sunday 23:59 UTC
    """
    weekday = dt.weekday()
    hour = dt.hour

    # Friday from 21:00 onwards
    if weekday == WEEKEND_START_DAY and hour >= WEEKEND_START_HOUR:
        return True

    # Saturday (all day)
    if weekday == 5:
        return True

    # Sunday (all day)
    if weekday == WEEKEND_END_DAY:
        return True

    return False


def get_weekend_start_date(dt):
    """
    For a datetime in a weekend window, return the Friday date that starts that weekend.
    This groups all weekend news by the Friday that starts the weekend.
    """
    weekday = dt.weekday()

    # If it's Friday 21:00+, Saturday, or Sunday - find the Friday
    if weekday == 4 and dt.hour >= WEEKEND_START_HOUR:
        # This Friday
        return dt.date()
    elif weekday == 5:
        # Saturday - go back to Friday
        return (dt - timedelta(days=1)).date()
    elif weekday == 6:
        # Sunday - go back to Friday
        return (dt - timedelta(days=2)).date()
    else:
        return None


def filter_weekend_news(df):
    """Filter news to only weekend articles and group by weekend"""
    logger.info("Filtering for weekend news...")

    # Ensure timestamp is UTC
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    # Filter to weekend times
    df['is_weekend'] = df['timestamp_utc'].apply(is_weekend_time)
    weekend_df = df[df['is_weekend']].copy()

    logger.info(f"Found {len(weekend_df)} weekend articles out of {len(df)} total")
    logger.info(f"Percentage: {len(weekend_df)/len(df)*100:.2f}%")

    # Assign weekend_start_date
    weekend_df['weekend_date'] = weekend_df['timestamp_utc'].apply(get_weekend_start_date)

    # Remove any null weekend_dates (shouldn't happen but be safe)
    weekend_df = weekend_df[weekend_df['weekend_date'].notna()]

    # Group statistics
    weekend_counts = weekend_df.groupby('weekend_date').size()
    logger.info(f"\nWeekend articles distribution:")
    logger.info(f"  Total weekends covered: {len(weekend_counts)}")
    logger.info(f"  Average articles per weekend: {weekend_counts.mean():.1f}")
    logger.info(f"  Median articles per weekend: {weekend_counts.median():.1f}")
    logger.info(f"  Min articles per weekend: {weekend_counts.min()}")
    logger.info(f"  Max articles per weekend: {weekend_counts.max()}")

    return weekend_df[['weekend_date', 'timestamp_utc', 'text', 'source']]


def main():
    """Main weekend filtering pipeline"""
    logger.info("="*60)
    logger.info("STEP 3: WEEKEND WINDOW FILTER")
    logger.info("="*60)
    logger.info(f"Weekend definition: Friday {WEEKEND_START_HOUR}:00 UTC - Sunday {WEEKEND_END_HOUR}:{WEEKEND_END_MINUTE} UTC")

    # Load combined news
    logger.info(f"\nLoading data from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please run news_loader.py first")
        return None

    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

    # Filter weekend news
    weekend_df = filter_weekend_news(df)

    # Save
    output_file = OUTPUT_DIR / "weekend_news.parquet"
    weekend_df.to_parquet(output_file, index=False)
    logger.info(f"\n✓ Saved {len(weekend_df)} weekend articles to {output_file}")

    # Show sample
    logger.info("\n" + "="*60)
    logger.info("SAMPLE WEEKEND ARTICLES")
    logger.info("="*60)
    for i, (weekend_date, group) in enumerate(weekend_df.groupby('weekend_date').head(3).groupby('weekend_date')):
        if i >= 3:  # Show first 3 weekends
            break
        logger.info(f"\nWeekend starting {weekend_date} ({len(group)} articles):")
        for _, row in group.head(2).iterrows():
            logger.info(f"  {row['timestamp_utc']} | {row['source']}")
            logger.info(f"    {row['text'][:100]}...")

    return weekend_df


if __name__ == "__main__":
    df = main()
