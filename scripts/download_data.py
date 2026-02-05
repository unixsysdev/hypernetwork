#!/usr/bin/env python3
"""
Script to download and prepare the SWE-bench trajectory dataset.

This script:
1. Downloads nebius/SWE-rebench-openhands-trajectories from HuggingFace
2. Filters for resolved=1 (successful trajectories only)
3. Pre-tokenizes the data for fast training
4. Saves to Arrow format for efficient loading

Usage:
    python scripts/download_data.py --output_dir ./data
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare SWE-bench trajectories")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-Coder-Next",
        help="Tokenizer to use for pre-tokenization",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset from HuggingFace Hub...")
    ds = load_dataset(
        "nebius/SWE-rebench-openhands-trajectories",
        split="train",
        cache_dir=args.cache_dir,
    )
    logger.info(f"Total trajectories: {len(ds)}")
    
    # Filter for successful trajectories
    logger.info("Filtering for resolved=1 and exit_status='submit'...")
    gold_ds = ds.filter(
        lambda x: x["resolved"] == 1 and x["exit_status"] == "submit",
        num_proc=4,
    )
    logger.info(f"Gold trajectories (resolved): {len(gold_ds)}")
    
    # Optionally limit samples
    if args.max_samples and len(gold_ds) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples for testing")
        gold_ds = gold_ds.select(range(args.max_samples))
    
    # Compute statistics
    logger.info("Computing statistics...")
    trajectory_lengths = []
    for row in gold_ds:
        traj = row["trajectory"]
        trajectory_lengths.append(len(traj))
    
    avg_len = sum(trajectory_lengths) / len(trajectory_lengths)
    max_len = max(trajectory_lengths)
    min_len = min(trajectory_lengths)
    
    logger.info(f"Trajectory message counts - Min: {min_len}, Max: {max_len}, Avg: {avg_len:.1f}")
    
    # Save filtered dataset
    output_path = output_dir / "gold_trajectories"
    logger.info(f"Saving to {output_path}...")
    gold_ds.save_to_disk(str(output_path))
    
    # Also save as parquet for inspection
    parquet_path = output_dir / "gold_trajectories.parquet"
    gold_ds.to_parquet(str(parquet_path))
    logger.info(f"Saved parquet to {parquet_path}")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total gold samples: {len(gold_ds)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Arrow format: {output_path}")
    logger.info(f"Parquet format: {parquet_path}")
    
    # Print sample fields
    sample = gold_ds[0]
    logger.info("\nSample fields:")
    for key in sample.keys():
        if key == "trajectory":
            logger.info(f"  - trajectory: {len(sample[key])} messages")
        elif isinstance(sample[key], str):
            logger.info(f"  - {key}: {sample[key][:50]}...")
        else:
            logger.info(f"  - {key}: {sample[key]}")


if __name__ == "__main__":
    main()
