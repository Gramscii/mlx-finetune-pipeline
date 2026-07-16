#!/usr/bin/env python3
"""Print the path of the adapter checkpoint with the lowest validation loss.
Used by run_pipeline.sh between SFT and fuse.

Usage:  python best_checkpoint.py logs/sft.log adapters/stage2-sft
"""
import sys
from pathlib import Path

from grpo_math import best_checkpoint

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: best_checkpoint.py <training_log> <adapter_dir>")
    print(best_checkpoint(Path(sys.argv[1]).read_text(), sys.argv[2]))
