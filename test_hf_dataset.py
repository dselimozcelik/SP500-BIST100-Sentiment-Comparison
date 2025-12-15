#!/usr/bin/env python3
"""Test loading a small sample from HuggingFace dataset"""
from datasets import load_dataset

# Test with a small, public dataset first
print("Testing HuggingFace dataset loading...")
print("-" * 50)

# Load a small sample from a public dataset (using imdb as an example)
print("\n1. Loading sample from 'imdb' dataset (small test)...")
try:
    dataset = load_dataset("imdb", split="train[:5]")  # Just 5 examples
    print(f"   ✓ Successfully loaded {len(dataset)} examples")
    print(f"   Features: {dataset.features}")
    print(f"\n   First example:")
    print(f"   Text (truncated): {dataset[0]['text'][:200]}...")
    print(f"   Label: {dataset[0]['label']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "-" * 50)
print("Dataset loading test complete!")
