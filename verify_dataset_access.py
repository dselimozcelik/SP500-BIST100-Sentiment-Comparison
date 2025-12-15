#!/usr/bin/env python3
"""Quick script to verify dataset access after accepting terms"""
from datasets import load_dataset

dataset_name = "Brianferrell787/financial-news-multisource"

print("Verifying dataset access...")
print("-" * 60)

try:
    # Try to load just one row with streaming
    dataset = load_dataset(
        dataset_name,
        data_files="data/sp500_daily_headlines/*.parquet",
        split="train",
        streaming=True
    )

    # Get first item
    first_item = next(iter(dataset))

    print("✓ SUCCESS! Dataset access granted.")
    print(f"\nColumns available: {list(first_item.keys())}")
    print("\nSample data:")
    for key, value in first_item.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

    print("\n" + "-" * 60)
    print("You can now run: python news_loader.py")

except Exception as e:
    print(f"✗ FAILED: {e}")
    print("\nPlease follow the instructions in DATASET_ACCESS_INSTRUCTIONS.md")

print("-" * 60)
