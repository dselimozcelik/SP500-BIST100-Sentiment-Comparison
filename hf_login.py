#!/usr/bin/env python3
"""Simple script to login to HuggingFace"""
from huggingface_hub import login
import sys

if len(sys.argv) > 1:
    token = sys.argv[1]
    login(token=token)
    print("Successfully logged in to HuggingFace!")
else:
    print("Usage: python hf_login.py <your_token>")
    print("\nOr set the token as an environment variable:")
    print("export HF_TOKEN=<your_token>")
    print("python hf_login.py")
