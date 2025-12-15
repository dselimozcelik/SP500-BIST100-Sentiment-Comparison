#!/usr/bin/env python3
"""
Step 1: Data Loading from HuggingFace dataset
Load financial news from multiple sources, parse timestamps, filter by date range,
and combine into a single dataframe.
"""
import pandas as pd
import json
from datasets import load_dataset
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "Brianferrell787/financial-news-multisource"
DATE_START = "2021-01-01"
DATE_END = "2025-12-31"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Subsets to load - focused on high-quality S&P 500 sources
# Avoiding fnspid_news (59 files, too large) and all_the_news_2 (general news)
SUBSETS = {
    "sp500_daily_headlines": "data/sp500_daily_headlines/*.parquet",
    "cnbc_headlines": "data/cnbc_headlines/*.parquet",
    "yahoo_finance_felixdrinkall": "data/yahoo_finance_felixdrinkall/*.parquet"
}


def parse_timestamp(row):
    """
    Extract timestamp from various possible fields.
    Tries: extra_fields JSON, direct timestamp columns, date columns
    """
    # Try extra_fields first
    if 'extra_fields' in row and row['extra_fields']:
        try:
            extra = json.loads(row['extra_fields']) if isinstance(row['extra_fields'], str) else row['extra_fields']

            # Common timestamp field names
            for field in ['published_utc', 'timestamp', 'date', 'datetime', 'published', 'created_at']:
                if field in extra and extra[field]:
                    try:
                        return pd.to_datetime(extra[field], utc=True)
                    except:
                        continue
        except:
            pass

    # Try direct timestamp columns
    for field in ['timestamp', 'date', 'published', 'datetime', 'created_at']:
        if field in row and row[field]:
            try:
                return pd.to_datetime(row[field], utc=True)
            except:
                continue

    return None


def extract_text(row):
    """Extract text/headline from various possible fields"""
    # Try common text fields
    for field in ['headline', 'title', 'text', 'content', 'description']:
        if field in row and row[field]:
            return str(row[field])
    return None


def load_subset(subset_name, data_files_pattern):
    """Load a single subset from the dataset"""
    logger.info(f"Loading {subset_name}...")

    try:
        # Load dataset with streaming to handle large files
        dataset = load_dataset(
            DATASET_NAME,
            data_files=data_files_pattern,
            split="train",
            streaming=True
        )

        records = []
        count = 0

        for row in dataset:
            # Extract timestamp
            timestamp = parse_timestamp(row)
            if timestamp is None:
                continue  # Skip rows without valid timestamp

            # Filter by date range (make sure to use UTC for comparison)
            if timestamp < pd.to_datetime(DATE_START, utc=True) or timestamp > pd.to_datetime(DATE_END, utc=True):
                continue

            # Extract text
            text = extract_text(row)
            if text is None or len(text.strip()) == 0:
                continue

            records.append({
                'timestamp_utc': timestamp,
                'text': text,
                'source': subset_name
            })

            count += 1
            if count % 10000 == 0:
                logger.info(f"  Processed {count} records from {subset_name}...")

        df = pd.DataFrame(records)
        logger.info(f"✓ Loaded {len(df)} records from {subset_name}")
        return df

    except Exception as e:
        logger.error(f"✗ Error loading {subset_name}: {e}")
        return pd.DataFrame(columns=['timestamp_utc', 'text', 'source'])


def main():
    """Main data loading pipeline"""
    logger.info("="*60)
    logger.info("STEP 1: DATA LOADING - S&P 500 FOCUSED")
    logger.info("="*60)
    logger.info(f"Date range: {DATE_START} to {DATE_END}")
    logger.info(f"Subsets: {list(SUBSETS.keys())}")
    logger.info("Note: Using high-quality S&P 500 sources only")

    # Load each subset
    all_dataframes = []
    for subset_name, data_files in SUBSETS.items():
        df = load_subset(subset_name, data_files)
        if len(df) > 0:
            all_dataframes.append(df)

    # Combine all dataframes
    if all_dataframes:
        logger.info("\nCombining all datasets...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp_utc').reset_index(drop=True)

        # Remove duplicates
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        logger.info(f"Removed {original_len - len(combined_df)} duplicate headlines")

        # Save to parquet
        output_file = OUTPUT_DIR / "combined_news.parquet"
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"\n✓ Saved {len(combined_df)} total records to {output_file}")

        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records: {len(combined_df):,}")
        logger.info(f"Date range: {combined_df['timestamp_utc'].min()} to {combined_df['timestamp_utc'].max()}")
        logger.info("\nRecords by source:")
        for source, count in combined_df['source'].value_counts().items():
            logger.info(f"  {source}: {count:,}")

        # Show sample
        logger.info("\nSample records:")
        print(combined_df.head(3).to_string())

        return combined_df
    else:
        logger.error("No data loaded!")
        return None


if __name__ == "__main__":
    df = main()
