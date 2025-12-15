"""
Module 1: Market Data Collection

Fetches daily OHLC data for S&P 500, calculates weekend gap returns.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLC data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with OHLC data
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data retrieved for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['Ticker'] = ticker
    data.index.name = 'Date'
    data = data.reset_index()

    return data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]


def calculate_weekend_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekend gap returns: (Monday_Open - Friday_Close) / Friday_Close

    Args:
        df: DataFrame with daily OHLC data

    Returns:
        DataFrame with weekend returns
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Add day of week (0=Monday, 4=Friday)
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    weekend_returns = []

    # Iterate through the dataframe to find Friday-Monday pairs
    for i in range(len(df) - 1):
        current_date = df.loc[i, 'Date']
        current_dow = df.loc[i, 'DayOfWeek']

        # Look for Fridays (day 4)
        if current_dow == 4:
            friday_close = df.loc[i, 'Close']
            friday_date = current_date

            # Find the next Monday (could be 3-4 days later due to holidays)
            # Search up to 10 days ahead to handle long holiday weekends
            for j in range(i + 1, min(i + 10, len(df))):
                next_date = df.loc[j, 'Date']
                next_dow = df.loc[j, 'DayOfWeek']
                days_diff = (next_date - current_date).days

                # Next Monday should be 3 days later (or more if there's a holiday)
                if next_dow == 0 and days_diff >= 3:
                    monday_open = df.loc[j, 'Open']
                    monday_date = next_date

                    # Calculate weekend gap return
                    weekend_return = (monday_open - friday_close) / friday_close

                    weekend_returns.append({
                        'friday_date': friday_date,
                        'monday_date': monday_date,
                        'friday_close': friday_close,
                        'monday_open': monday_open,
                        'weekend_return': weekend_return,
                        'ticker': df.loc[i, 'Ticker']
                    })
                    break

    return pd.DataFrame(weekend_returns)


def collect_all_market_data(start_date: str = "2020-01-01",
                            end_date: str = "2024-12-31",
                            output_dir: str = "data") -> dict:
    """
    Collect market data for S&P 500, calculate weekend returns.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        output_dir: Directory to save parquet files

    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    market_name = 'SP500'
    ticker = '^GSPC'

    print(f"\n{'='*60}")
    print(f"Processing {market_name} ({ticker})")
    print(f"{'='*60}")

    # Fetch daily data
    daily_data = fetch_market_data(ticker, start_date, end_date)
    print(f"Retrieved {len(daily_data)} daily records")

    # Calculate weekend returns
    weekend_data = calculate_weekend_returns(daily_data)
    print(f"Calculated {len(weekend_data)} weekend returns")

    # Save to parquet
    daily_file = output_path / f"{market_name}_daily.parquet"
    weekend_file = output_path / f"{market_name}_weekend_returns.parquet"

    daily_data.to_parquet(daily_file, index=False)
    weekend_data.to_parquet(weekend_file, index=False)

    print(f"Saved daily data to: {daily_file}")
    print(f"Saved weekend returns to: {weekend_file}")

    results = {
        market_name: {
            'daily_file': daily_file,
            'weekend_file': weekend_file,
            'daily_records': len(daily_data),
            'weekend_records': len(weekend_data)
        }
    }

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{market_name}:")
    print(f"  Daily records: {results[market_name]['daily_records']}")
    print(f"  Weekend returns: {results[market_name]['weekend_records']}")
    print(f"  Date range: {start_date} to {end_date}")

    return results


if __name__ == "__main__":
    # Run data collection
    results = collect_all_market_data(
        start_date="2020-01-01",
        end_date="2024-12-31"
    )

    # Display sample of weekend returns
    print(f"\n{'='*60}")
    print("SAMPLE WEEKEND RETURNS")
    print(f"{'='*60}")

    df = pd.read_parquet(results['SP500']['weekend_file'])
    print(f"\nSP500 (first 5 weekends):")
    print(df.head())
    print(f"\nDescriptive statistics:")
    print(df['weekend_return'].describe())
