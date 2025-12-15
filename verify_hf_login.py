#!/usr/bin/env python3
"""Verify HuggingFace login and test access to the financial news dataset"""
from datasets import load_dataset
from huggingface_hub import whoami

print("Verifying HuggingFace authentication...")
print("-" * 60)

# Check who is logged in
try:
    user_info = whoami()
    print(f"✓ Logged in as: {user_info['name']}")
except Exception as e:
    print(f"✗ Login verification failed: {e}")
    exit(1)

# Try to access the financial news dataset
print("\nTesting access to 'Brianferrell787/financial-news-multisource'...")
try:
    # Try loading just the dataset info first (no actual data download)
    print("  Loading dataset info...")
    dataset = load_dataset(
        "Brianferrell787/financial-news-multisource",
        data_files="data/sp500_daily_headlines/*.parquet",
        split="train[:2]"  # Just 2 examples to test
    )
    print(f"✓ Successfully accessed dataset!")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Sample row: {dataset[0]}")

except Exception as e:
    print(f"✗ Dataset access failed: {e}")
    print("\nNote: You may need to:")
    print("  1. Accept the dataset terms on HuggingFace website")
    print("  2. Verify the dataset name and path are correct")
    exit(1)

print("\n" + "-" * 60)
print("HuggingFace access verified successfully!")
