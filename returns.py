#!/usr/bin/env python3
"""
Step 5: S&P 500 Returns Calculation
Calculate Monday gap returns (Monday Open - Friday Close) / Friday Close
Match returns to weekend dates for regression analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yfinance as yf
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SENTIMENT_FILE = Path("data/weekend_sentiment.parquet")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# S&P 500 ticker
SP500_TICKER = "^GSPC"
DATE_START = "2021-01-01"
DATE_END = "2025-12-31"


def download_sp500_data():
    """Download S&P 500 historical data"""
    logger.info(f"Downloading S&P 500 data from {DATE_START} to {DATE_END}...")

    sp500 = yf.Ticker(SP500_TICKER)
    df = sp500.history(start=DATE_START, end=DATE_END)

    logger.info(f"✓ Downloaded {len(df)} trading days")

    # Keep only what we need
    df = df[['Open', 'Close']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone for easier matching

    return df


def calculate_monday_gap_returns(sp500_df):
    """
    Calculate Monday gap returns: (Monday_Open - Friday_Close) / Friday_Close
    This measures the overnight/weekend price change.
    """
    logger.info("Calculating Monday gap returns...")

    # Add day of week
    sp500_df['weekday'] = sp500_df.index.dayofweek

    # Identify Mondays (0 = Monday)
    mondays = sp500_df[sp500_df['weekday'] == 0].copy()
    logger.info(f"Found {len(mondays)} Mondays")

    # For each Monday, find the previous Friday's close
    returns_data = []

    for monday_date, monday_row in mondays.iterrows():
        # Look back up to 5 days to find Friday (handles holidays)
        friday_close = None
        friday_date = None

        for days_back in range(1, 6):
            potential_friday = monday_date - timedelta(days=days_back)

            if potential_friday in sp500_df.index:
                # Check if it's actually a Friday (or closest trading day before weekend)
                potential_weekday = potential_friday.weekday()

                # Accept Friday (4) or Thursday (3) if Friday was a holiday
                if potential_weekday == 4:
                    friday_close = sp500_df.loc[potential_friday, 'Close']
                    friday_date = potential_friday.date()
                    break
                elif potential_weekday == 3 and days_back >= 2:
                    # Thursday, but only if we've checked Friday and it wasn't there
                    friday_close = sp500_df.loc[potential_friday, 'Close']
                    friday_date = potential_friday.date()
                    break

        if friday_close is not None:
            monday_open = monday_row['Open']

            # Calculate gap return
            gap_return = (monday_open - friday_close) / friday_close

            returns_data.append({
                'weekend_date': friday_date,
                'friday_close': friday_close,
                'monday_date': monday_date.date(),
                'monday_open': monday_open,
                'gap_return': gap_return,
                'gap_return_pct': gap_return * 100
            })

    returns_df = pd.DataFrame(returns_data)
    logger.info(f"✓ Calculated {len(returns_df)} weekend gap returns")

    return returns_df


def merge_sentiment_and_returns(sentiment_df, returns_df):
    """Merge sentiment data with returns data"""
    logger.info("Merging sentiment and returns...")

    # Convert weekend_date to date type for matching
    sentiment_df['weekend_date'] = pd.to_datetime(sentiment_df['weekend_date']).dt.date
    returns_df['weekend_date'] = pd.to_datetime(returns_df['weekend_date']).dt.date

    # Merge on weekend_date
    merged = pd.merge(
        sentiment_df,
        returns_df,
        on='weekend_date',
        how='inner'
    )

    logger.info(f"✓ Matched {len(merged)} weekends with both sentiment and returns")
    logger.info(f"  Lost {len(sentiment_df) - len(merged)} weekends due to missing market data")

    return merged


def main():
    """Main returns calculation pipeline"""
    logger.info("="*60)
    logger.info("STEP 5: S&P 500 RETURNS CALCULATION")
    logger.info("="*60)

    # Load weekend sentiment
    logger.info(f"\nLoading sentiment data from {SENTIMENT_FILE}...")
    if not SENTIMENT_FILE.exists():
        logger.error(f"Input file not found: {SENTIMENT_FILE}")
        logger.error("Please run sentiment.py first")
        return None

    sentiment_df = pd.read_parquet(SENTIMENT_FILE)
    logger.info(f"Loaded sentiment for {len(sentiment_df)} weekends")

    # Download S&P 500 data
    sp500_df = download_sp500_data()

    # Calculate Monday gap returns
    returns_df = calculate_monday_gap_returns(sp500_df)

    # Merge sentiment and returns
    merged_df = merge_sentiment_and_returns(sentiment_df, returns_df)

    # Save merged dataset
    output_file = OUTPUT_DIR / "sentiment_returns.parquet"
    merged_df.to_parquet(output_file, index=False)
    logger.info(f"\n✓ Saved merged data to {output_file}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("RETURNS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total weekends with returns: {len(merged_df)}")
    logger.info(f"Date range: {merged_df['weekend_date'].min()} to {merged_df['weekend_date'].max()}")
    logger.info(f"\nGap Return Statistics:")
    logger.info(f"  Mean: {merged_df['gap_return_pct'].mean():.3f}%")
    logger.info(f"  Std:  {merged_df['gap_return_pct'].std():.3f}%")
    logger.info(f"  Min:  {merged_df['gap_return_pct'].min():.3f}%")
    logger.info(f"  Max:  {merged_df['gap_return_pct'].max():.3f}%")
    logger.info(f"\nPositive returns: {(merged_df['gap_return'] > 0).sum()} ({(merged_df['gap_return'] > 0).mean()*100:.1f}%)")
    logger.info(f"Negative returns: {(merged_df['gap_return'] < 0).sum()} ({(merged_df['gap_return'] < 0).mean()*100:.1f}%)")

    # Show correlation preview
    correlation = merged_df['sentiment_composite'].corr(merged_df['gap_return'])
    logger.info(f"\nCorrelation (Sentiment vs Returns): {correlation:.4f}")

    # Show sample
    logger.info("\n" + "="*60)
    logger.info("SAMPLE DATA")
    logger.info("="*60)
    for _, row in merged_df.head(5).iterrows():
        logger.info(f"\nWeekend: {row['weekend_date']} -> Monday: {row['monday_date']}")
        logger.info(f"  Articles: {row['article_count']}")
        logger.info(f"  Sentiment: {row['sentiment_composite']:.3f}")
        logger.info(f"  Gap Return: {row['gap_return_pct']:.3f}%")

    return merged_df


if __name__ == "__main__":
    df = main()
