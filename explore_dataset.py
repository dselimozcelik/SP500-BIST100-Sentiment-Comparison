#!/usr/bin/env python3
"""Explore the financial news dataset structure"""
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import list_repo_files

dataset_name = "Brianferrell787/financial-news-multisource"

print(f"Exploring dataset: {dataset_name}")
print("-" * 60)

# Try to list files in the repository
print("\n1. Listing files in repository...")
try:
    files = list(list_repo_files(dataset_name, repo_type="dataset"))
    print(f"   Found {len(files)} files:")
    for f in sorted(files)[:30]:  # Show first 30 files
        print(f"   - {f}")
    if len(files) > 30:
        print(f"   ... and {len(files) - 30} more files")
except Exception as e:
    print(f"   Error listing files: {e}")

# Try to get dataset configs
print("\n2. Checking dataset configurations...")
try:
    configs = get_dataset_config_names(dataset_name)
    print(f"   Available configs: {configs}")
except Exception as e:
    print(f"   Error getting configs: {e}")

# Try loading without specifying data_files
print("\n3. Attempting to load dataset without data_files...")
try:
    dataset = load_dataset(dataset_name, split="train[:2]")
    print(f"   ✓ Success! Columns: {dataset.column_names}")
    print(f"   Sample: {dataset[0]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "-" * 60)
