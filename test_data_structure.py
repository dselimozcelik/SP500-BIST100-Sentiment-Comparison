#!/usr/bin/env python3
"""Test loading a sample to understand data structure"""
from datasets import load_dataset
import json

dataset_name = "Brianferrell787/financial-news-multisource"

print("Testing data structure...")
print("-" * 60)

# Test each subset to understand the schema
subsets = [
    "sp500_daily_headlines",
    "cnbc_headlines",
]

for subset in subsets:
    print(f"\n{subset}:")
    print("-" * 40)
    try:
        data = load_dataset(
            dataset_name,
            data_files=f"data/{subset}/*.parquet",
            split="train[:2]"
        )
        print(f"Columns: {data.column_names}")
        print(f"\nFirst row:")
        first_row = data[0]
        for key, value in first_row.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")

        # Check if extra_fields exists and try to parse it
        if 'extra_fields' in first_row:
            print(f"\n  Parsed extra_fields:")
            try:
                extra = json.loads(first_row['extra_fields'])
                for k, v in extra.items():
                    print(f"    {k}: {v}")
            except:
                print(f"    Could not parse JSON")

    except Exception as e:
        print(f"Error loading {subset}: {e}")

print("\n" + "-" * 60)
