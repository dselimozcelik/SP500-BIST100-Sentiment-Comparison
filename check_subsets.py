#!/usr/bin/env python3
"""Check which requested subsets are available"""
from huggingface_hub import list_repo_files

dataset_name = "Brianferrell787/financial-news-multisource"
requested_subsets = ["sp500_daily_headlines", "fnspid_news", "cnbc_headlines", "all_the_news_2"]

print("Checking for requested subsets...")
print("-" * 60)

files = list(list_repo_files(dataset_name, repo_type="dataset"))

for subset in requested_subsets:
    matching_files = [f for f in files if subset in f and f.endswith('.parquet')]
    if matching_files:
        print(f"\n✓ {subset}: FOUND ({len(matching_files)} files)")
        for f in matching_files[:3]:
            print(f"  - {f}")
        if len(matching_files) > 3:
            print(f"  ... and {len(matching_files) - 3} more")
    else:
        print(f"\n✗ {subset}: NOT FOUND")

print("\n" + "-" * 60)
print("\nAll available subsets:")
subsets = set()
for f in files:
    if f.startswith('data/') and '/' in f[5:]:
        subset = f.split('/')[1]
        subsets.add(subset)

for subset in sorted(subsets):
    count = len([f for f in files if f'data/{subset}/' in f and f.endswith('.parquet')])
    print(f"  - {subset} ({count} files)")
