#!/usr/bin/env python3
"""
Pre-compute Teacher Logits using vLLM for 10-20x Speedup.

vLLM provides:
- Continuous batching for high throughput
- PagedAttention for efficient memory
- FP8 quantization support
- Tensor parallelism across GPUs

Expected performance:
- transformers: ~2-3 samples/minute on 8xH200
- vLLM: ~30-60 samples/minute on 8xH200

Usage:
    # Production run (4 GPUs for Teacher, other 4 for Student training later)
    python scripts/cache_teacher_vllm.py --output_dir ./teacher_cache --tp 4
    
    # Maximum throughput (all 8 GPUs for caching, ~2x faster)
    python scripts/cache_teacher_vllm.py --output_dir ./teacher_cache --tp 8
    
    # Test mode
    python scripts/cache_teacher_vllm.py --output_dir ./teacher_cache --test --max_samples 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_top_k_logits(
    logits: np.ndarray,
    k: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-K logits for efficient storage."""
    # logits: [seq_len, vocab_size]
    indices = np.argpartition(logits, -k, axis=-1)[..., -k:]
    # Sort by value within top-k
    sorted_indices = np.argsort(
        np.take_along_axis(logits, indices, axis=-1),
        axis=-1
    )[..., ::-1]
    indices = np.take_along_axis(indices, sorted_indices, axis=-1)
    values = np.take_along_axis(logits, indices, axis=-1)
    
    return (
        values.astype(np.float16),
        indices.astype(np.int32),
    )


def build_trajectory_text(trajectory: list) -> str:
    """Build trajectory text from messages."""
    if not trajectory:
        return ""
    
    parts = []
    for msg in trajectory:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            parts.append(f"[{role}]: {content}")
    
    return "\n\n".join(parts)


def cache_with_vllm(
    output_dir: Path,
    dataset,
    teacher_model: str,
    tensor_parallel_size: int = 4,
    max_tokens: int = 8192,
    top_k: int = 128,
    gpu_memory_utilization: float = 0.9,
):
    """Cache teacher logits using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        raise
    
    logger.info(f"Loading Teacher with vLLM: {teacher_model}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    
    # Check for FP8 support (H200, H100, etc.)
    fp8_enabled = False
    try:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 9:  # Hopper+
                fp8_enabled = True
                logger.info("FP8 quantization: ENABLED (Hopper GPU detected)")
    except Exception:
        pass
    
    # Initialize vLLM
    llm = LLM(
        model=teacher_model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        quantization="fp8" if fp8_enabled else None,
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Process in batches for efficiency
    batch_size = 8  # Adjust based on GPU memory
    trajectories = []
    indices = []
    
    for idx, row in enumerate(dataset):
        output_path = output_dir / f"trajectory_{idx:06d}.npz"
        if output_path.exists():
            continue
        
        trajectory_text = build_trajectory_text(row["trajectory"])
        if len(trajectory_text) < 100:
            continue
        
        trajectories.append(trajectory_text)
        indices.append(idx)
        
        # Process batch
        if len(trajectories) >= batch_size:
            process_batch(
                llm, tokenizer, trajectories, indices, 
                output_dir, max_tokens, top_k
            )
            trajectories = []
            indices = []
    
    # Process remaining
    if trajectories:
        process_batch(
            llm, tokenizer, trajectories, indices,
            output_dir, max_tokens, top_k
        )


def process_batch(
    llm,
    tokenizer,
    texts: List[str],
    indices: List[int],
    output_dir: Path,
    max_tokens: int,
    top_k: int,
):
    """Process a batch of trajectories with vLLM."""
    from vllm import SamplingParams
    
    # Tokenize all texts
    all_input_ids = []
    for text in texts:
        input_ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        all_input_ids.append(input_ids)
    
    # Use vLLM's prompt_logprobs feature to get logits for all positions
    # Note: This uses generate with max_new_tokens=0 to get logprobs only
    sampling_params = SamplingParams(
        max_tokens=1,  # Generate 1 token to get logprobs
        prompt_logprobs=1,  # Get logprobs for all prompt tokens
        logprobs=top_k,  # Get top-k logprobs
    )
    
    # Generate to get logprobs
    outputs = llm.generate(
        prompt_token_ids=all_input_ids,
        sampling_params=sampling_params,
    )
    
    # Extract and save logprobs
    for output, idx, input_ids in zip(outputs, indices, all_input_ids):
        output_path = output_dir / f"trajectory_{idx:06d}.npz"
        
        try:
            # Extract prompt logprobs (for all input positions)
            prompt_logprobs = output.prompt_logprobs
            
            if prompt_logprobs is None:
                logger.warning(f"No logprobs for trajectory {idx}")
                continue
            
            # Convert logprobs to arrays
            seq_len = len(prompt_logprobs)
            values = np.zeros((seq_len, top_k), dtype=np.float16)
            token_indices = np.zeros((seq_len, top_k), dtype=np.int32)
            
            for pos, logprob_dict in enumerate(prompt_logprobs):
                if logprob_dict is None:
                    continue
                
                # Sort by logprob value
                sorted_items = sorted(
                    logprob_dict.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True
                )[:top_k]
                
                for k_idx, (token_id, logprob_obj) in enumerate(sorted_items):
                    if k_idx >= top_k:
                        break
                    values[pos, k_idx] = logprob_obj.logprob
                    token_indices[pos, k_idx] = token_id
            
            # Save with format marker
            np.savez_compressed(
                output_path,
                values=values,
                indices=token_indices,
                input_ids=np.array(input_ids, dtype=np.int32),
                format=np.array(['logprobs']),  # Mark as log-probs from vLLM
            )
            
        except Exception as e:
            logger.warning(f"Failed on trajectory {idx}: {e}")
            continue


def cache_with_transformers(
    output_dir: Path,
    dataset,
    teacher_model: str,
    max_tokens: int = 8192,
    top_k: int = 128,
):
    """Fallback: Cache using transformers (slower)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading Teacher with transformers: {teacher_model}")
    logger.warning("Using transformers is 10-20x slower than vLLM!")
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    for idx, row in enumerate(tqdm(dataset)):
        output_path = output_dir / f"trajectory_{idx:06d}.npz"
        if output_path.exists():
            continue
        
        trajectory_text = build_trajectory_text(row["trajectory"])
        if len(trajectory_text) < 100:
            continue
        
        try:
            inputs = tokenizer(
                trajectory_text,
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0].cpu().numpy()
            
            values, indices = compute_top_k_logits(logits, k=top_k)
            
            np.savez_compressed(
                output_path,
                values=values,
                indices=indices,
                input_ids=input_ids.cpu().numpy().astype(np.int32),
                format=np.array(['logits']),  # Transformers gives raw logits
            )
            
        except Exception as e:
            logger.warning(f"Failed on trajectory {idx}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Cache teacher logits with vLLM")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-Coder-480B-A35B-Instruct")
    parser.add_argument("--data_dir", type=str, default="./data/gold_trajectories")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test mode with mock data")
    parser.add_argument("--use_transformers", action="store_true", help="Use transformers instead of vLLM")
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
    
    # Choose engine
    if args.test:
        logger.info("Test mode: generating mock data")
        for idx in range(len(dataset)):
            output_path = output_dir / f"trajectory_{idx:06d}.npz"
            if output_path.exists():
                continue
            
            seq_len = 512
            np.savez_compressed(
                output_path,
                values=np.random.randn(seq_len, args.top_k).astype(np.float16),
                indices=np.random.randint(0, 150000, (seq_len, args.top_k), dtype=np.int32),
                input_ids=np.random.randint(0, 150000, (1, seq_len), dtype=np.int32),
            )
    elif args.use_transformers:
        cache_with_transformers(
            output_dir, dataset, args.teacher_model,
            args.max_tokens, args.top_k
        )
    else:
        cache_with_vllm(
            output_dir, dataset, args.teacher_model,
            tensor_parallel_size=args.tp,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
        )
    
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
