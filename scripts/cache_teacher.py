#!/usr/bin/env python3
"""
Pre-compute Teacher Logits for Offline Distillation.

This script generates and caches teacher logits BEFORE training begins.
This decouples the expensive teacher inference from the training loop.

Benefits:
1. Training becomes pure gradient computation (fast)
2. No need for vLLM integration during training
3. Can use cheap cloud GPUs for one-time computation
4. ~125GB storage for 20k trajectories

Storage calculation:
    20,000 trajectories × 8,192 tokens × 128 top-k × 6 bytes = ~125 GB
    
Where 6 bytes = 2 (float16 logit) + 4 (int32 index)

Usage:
    # Full dataset (requires 8x GPUs with 480B model)
    python scripts/cache_teacher.py --output_dir ./teacher_cache
    
    # Test mode (small sample)
    python scripts/cache_teacher.py --output_dir ./teacher_cache --test --max_samples 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_top_k_logits(
    logits: torch.Tensor,
    k: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract top-K logits for efficient storage.
    
    Args:
        logits: [seq_len, vocab_size] - Full logits
        k: Number of top logits to keep
        
    Returns:
        values: [seq_len, k] float16 - Top-K logit values
        indices: [seq_len, k] int32 - Top-K vocabulary indices
    """
    values, indices = torch.topk(logits, k, dim=-1)
    return (
        values.cpu().numpy().astype(np.float16),
        indices.cpu().numpy().astype(np.int32),
    )


def cache_single_trajectory(
    teacher,
    tokenizer,
    trajectory_text: str,
    output_path: Path,
    max_tokens: int = 8192,
    top_k: int = 128,
):
    """
    Compute and cache teacher logits for a single trajectory.
    
    Args:
        teacher: The teacher model
        tokenizer: Tokenizer
        trajectory_text: Full trajectory as text
        output_path: Where to save the cached logits
        max_tokens: Maximum context length
        top_k: Number of top logits to cache
    """
    # Tokenize
    inputs = tokenizer(
        trajectory_text,
        max_length=max_tokens,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(teacher.device)
    
    # Forward pass (no grad needed)
    with torch.no_grad():
        outputs = teacher(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab]
    
    # Extract top-K
    values, indices = compute_top_k_logits(logits, k=top_k)
    
    # Save compressed with format marker
    np.savez_compressed(
        output_path,
        values=values,
        indices=indices,
        input_ids=input_ids.cpu().numpy().astype(np.int32),
        format=np.array(['logits']),  # Mark as raw logits (not log-probs)
    )


def load_cached_logits(cache_path: Path) -> dict:
    """Load cached logits for training."""
    data = np.load(cache_path)
    return {
        "values": torch.from_numpy(data["values"].astype(np.float32)),
        "indices": torch.from_numpy(data["indices"].astype(np.int64)),
        "input_ids": torch.from_numpy(data["input_ids"].astype(np.int64)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cache teacher logits")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-Coder-480B-A35B-Instruct")
    parser.add_argument("--data_dir", type=str, default="./data/gold_trajectories")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test mode with mock teacher")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    from datasets import load_from_disk
    logger.info(f"Loading dataset from {args.data_dir}")
    
    try:
        dataset = load_from_disk(args.data_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Run scripts/download_data.py first to prepare the dataset")
        return 1
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"Processing {len(dataset)} trajectories")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    
    if args.test:
        # Mock teacher for testing
        logger.info("Test mode: using mock teacher")
        
        class MockTeacher:
            device = "cpu"
            
            def __call__(self, input_ids):
                # Return random logits
                seq_len = input_ids.shape[1]
                vocab_size = 150000
                
                class Output:
                    logits = torch.randn(1, seq_len, vocab_size)
                
                return Output()
        
        teacher = MockTeacher()
    else:
        # Load real teacher
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading teacher model: {args.teacher_model}")
        logger.info("This may take a while for 480B parameters...")
        
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        teacher.eval()
    
    # Process trajectories
    logger.info("Starting teacher inference...")
    
    for idx, row in enumerate(tqdm(dataset)):
        output_path = output_dir / f"trajectory_{idx:06d}.npz"
        
        if output_path.exists():
            continue  # Skip already cached
        
        # Build trajectory text
        trajectory = row["trajectory"]
        if not trajectory:
            continue
        
        # Simple concatenation (you might want to use chat template)
        trajectory_text = ""
        for msg in trajectory:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                trajectory_text += f"[{role}]: {content}\n\n"
        
        if len(trajectory_text) < 100:
            continue
        
        try:
            cache_single_trajectory(
                teacher,
                tokenizer,
                trajectory_text,
                output_path,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
            )
        except Exception as e:
            logger.warning(f"Failed on trajectory {idx}: {e}")
            continue
    
    # Summary
    cached_files = list(output_dir.glob("*.npz"))
    total_size = sum(f.stat().st_size for f in cached_files)
    
    logger.info("\n" + "=" * 50)
    logger.info("CACHING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Cached trajectories: {len(cached_files)}")
    logger.info(f"Total size: {total_size / 1e9:.2f} GB")
    logger.info(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
