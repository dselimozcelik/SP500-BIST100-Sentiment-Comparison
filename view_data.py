"""
Simple data viewer for exploring collected news and market data
"""

import pandas as pd
from pathlib import Path
import sys

def view_news(market='SP500', num_rows=10):
    """View news articles for a specific market"""
    file_path = f"data/{market}_news.parquet"

    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_parquet(file_path)

    print(f"\n{'='*80}")
    print(f"{market} NEWS DATA")
    print(f"{'='*80}")
    print(f"Total articles: {len(df)}")
    print(f"Date range: {df['friday_date'].min()} to {df['friday_date'].max()}")
    print(f"Unique weekends: {df['friday_date'].nunique()}")
    print(f"\nColumns: {list(df.columns)}")

    print(f"\n{'='*80}")
    print(f"FIRST {num_rows} ARTICLES")
    print(f"{'='*80}\n")

    # Display articles with better formatting
    for idx, row in df.head(num_rows).iterrows():
        print(f"[{idx+1}] {row['friday_date']}")
        print(f"    Title: {row['title']}")
        print(f"    Source: {row['source']}")
        print(f"    Query: {row['query']}")
        print(f"    URL: {row['url'][:80]}...")
        print()

def view_weekend_returns(market='SP500', num_rows=10):
    """View weekend returns for a specific market"""
    file_path = f"data/{market}_weekend_returns.parquet"

    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_parquet(file_path)

    print(f"\n{'='*80}")
    print(f"{market} WEEKEND RETURNS")
    print(f"{'='*80}")
    print(f"Total weekends: {len(df)}")
    print(f"\nSummary statistics:")
    print(df['weekend_return'].describe())

    print(f"\n{'='*80}")
    print(f"FIRST {num_rows} WEEKENDS")
    print(f"{'='*80}\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(df.head(num_rows).to_string(index=False))

def export_to_csv(file_type='news', market='SP500'):
    """Export parquet to CSV for viewing in Excel/spreadsheet"""
    if file_type == 'news':
        input_file = f"data/{market}_news.parquet"
        output_file = f"data/{market}_news.csv"
    else:
        input_file = f"data/{market}_weekend_returns.parquet"
        output_file = f"data/{market}_weekend_returns.csv"

    if not Path(input_file).exists():
        print(f"File not found: {input_file}")
        return

    df = pd.read_parquet(input_file)
    df.to_csv(output_file, index=False)
    print(f"✓ Exported to {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  You can open this in Excel or any text editor")

def search_news(keyword, market='SP500'):
    """Search for specific keywords in news titles"""
    file_path = f"data/{market}_news.parquet"

    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_parquet(file_path)

    # Case-insensitive search in title
    matches = df[df['title'].str.contains(keyword, case=False, na=False)]

    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS: '{keyword}' in {market} news")
    print(f"{'='*80}")
    print(f"Found {len(matches)} matches out of {len(df)} total articles\n")

    for idx, row in matches.head(20).iterrows():
        print(f"[{idx+1}] {row['friday_date']}")
        print(f"    {row['title']}")
        print(f"    {row['source']}")
        print()

if __name__ == "__main__":
    # Interactive menu
    print("\n" + "="*80)
    print("DATA VIEWER - Weekend News Sentiment Analysis")
    print("="*80)
    print("\nAvailable commands:")
    print("  1. View S&P 500 news")
    print("  2. View S&P 500 weekend returns")
    print("  3. Export S&P 500 news to CSV")
    print("  4. Search S&P 500 news")
    print("  0. Exit")

    while True:
        choice = input("\nEnter choice (0-4): ").strip()

        if choice == '1':
            num = input("How many articles to show? (default 10): ").strip() or "10"
            view_news('SP500', int(num))

        elif choice == '2':
            num = input("How many weekends to show? (default 10): ").strip() or "10"
            view_weekend_returns('SP500', int(num))

        elif choice == '3':
            export_to_csv('news', 'SP500')

        elif choice == '4':
            keyword = input("Search keyword: ").strip()
            search_news(keyword, 'SP500')

        elif choice == '0':
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
