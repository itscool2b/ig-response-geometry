"""Explicit inputs and non-overwriting outputs for optional primitive demos."""
import argparse
from pathlib import Path


def parse_demo_args(kind):
    parser = argparse.ArgumentParser(description="Optional primitive IG demonstration; current GPU execution is not certified by historical figures.")
    if kind == "vision":
        parser.add_argument("--image", type=Path, required=True, help="An image the caller is authorized to use")
    else:
        parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True, help="New PNG path; historical figures are never overwritten")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.m < 1:
        parser.error("m must be positive")
    if args.out.exists():
        parser.error("Output exists; choose a fresh filename")
    if args.out.suffix.lower() != ".png":
        parser.error("Output must be PNG")
    return args
