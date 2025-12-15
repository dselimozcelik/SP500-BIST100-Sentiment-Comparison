#!/usr/bin/env python3
"""Test loading with proper configuration"""
from datasets import load_dataset
import json

dataset_name = "Brianferrell787/financial-news-multisource"

print("Testing data loading with different approaches...")
print("-" * 60)

# Approach 1: Load with explicit paths
print("\nApproach 1: Loading with explicit file paths...")
try:
    data = load_dataset(
        dataset_name,
        data_files={
            "sp500": "data/sp500_daily_headlines/sp500_daily_headlines.000.parquet"
        }
    )
    print(f"✓ Success!")
    print(f"  Columns: {data['sp500'].column_names}")
    print(f"  Sample: {data['sp500'][0]}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Approach 2: Try with streaming
print("\nApproach 2: Loading with streaming...")
try:
    data = load_dataset(
        dataset_name,
        data_files="data/sp500_daily_headlines/*.parquet",
        streaming=True
    )
    print(f"✓ Success with streaming!")
    # Get first item from iterator
    first_item = next(iter(data['train']))
    print(f"  Columns: {list(first_item.keys())}")
    for key, value in first_item.items():
        if isinstance(value, str) and len(value) > 150:
            print(f"  {key}: {value[:150]}...")
        else:
            print(f"  {key}: {value}")

    # Try to parse extra_fields if it exists
    if 'extra_fields' in first_item:
        print(f"\n  Parsed extra_fields:")
        try:
            extra = json.loads(first_item['extra_fields'])
            for k, v in list(extra.items())[:10]:  # First 10 fields
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"    Could not parse: {e}")

except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "-" * 60)
