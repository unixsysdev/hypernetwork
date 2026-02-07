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
from typing import Optional, List, Tuple

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


def build_trajectory_text(trajectory: list) -> Tuple[str, List[Tuple[int, int]], int]:
    """
    Build trajectory text from messages, tracking role boundaries.
    
    Returns:
        text: The full trajectory text
        assistant_ranges: List of (start_char, end_char) for assistant content
        prompt_boundary_char: Character offset where first assistant turn starts
    """
    if not trajectory:
        return "", [], 0
    
    parts = []
    assistant_ranges = []
    prompt_boundary_char = -1
    current_pos = 0
    
    for i, msg in enumerate(trajectory):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        
        # Add separator between parts
        if parts:
            current_pos += 2  # "\n\n" separator
        
        part = f"[{role}]: {content}"
        part_start = current_pos
        part_end = current_pos + len(part)
        
        if role == "assistant":
            assistant_ranges.append((part_start, part_end))
            if prompt_boundary_char == -1:
                prompt_boundary_char = part_start
        
        parts.append(part)
        current_pos = part_end
    
    # If no assistant turns found, boundary is at the end
    if prompt_boundary_char == -1:
        prompt_boundary_char = current_pos
    
    return "\n\n".join(parts), assistant_ranges, prompt_boundary_char


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
        
        trajectory_text, assistant_ranges, prompt_boundary_char = build_trajectory_text(row["trajectory"])
        if len(trajectory_text) < 100:
            continue
        
        trajectories.append((trajectory_text, assistant_ranges, prompt_boundary_char))
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


def _build_loss_mask(
    text: str,
    assistant_ranges: List[Tuple[int, int]],
    prompt_boundary_char: int,
    tokenizer,
    input_ids: List[int],
    max_tokens: int,
) -> Tuple[np.ndarray, int]:
    """
    Build a binary loss mask (1=assistant, 0=other) and prompt boundary token index.
    
    Maps character-level assistant ranges to token positions using offset mapping.
    """
    # Get offset mapping: list of (char_start, char_end) per token
    encoding = tokenizer(text, truncation=True, max_length=max_tokens, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]  # [(start, end), ...]
    seq_len = len(input_ids)
    
    loss_mask = np.zeros(seq_len, dtype=np.int32)
    prompt_boundary_token = seq_len  # default: entire sequence is prompt
    
    for tok_idx in range(seq_len):
        if tok_idx >= len(offsets):
            break
        tok_start, tok_end = offsets[tok_idx]
        if tok_end == 0:
            continue  # special token with no character mapping
        
        # Check if this token overlaps any assistant range
        for a_start, a_end in assistant_ranges:
            if tok_start < a_end and tok_end > a_start:
                loss_mask[tok_idx] = 1
                break
    
    # Find prompt boundary token: first token whose start >= prompt_boundary_char
    for tok_idx in range(seq_len):
        if tok_idx >= len(offsets):
            break
        tok_start, _ = offsets[tok_idx]
        if tok_start >= prompt_boundary_char:
            prompt_boundary_token = tok_idx
            break
    
    return loss_mask, prompt_boundary_token


def process_batch(
    llm,
    tokenizer,
    trajectory_items: List[Tuple[str, List[Tuple[int, int]], int]],
    indices: List[int],
    output_dir: Path,
    max_tokens: int,
    top_k: int,
):
    """Process a batch of trajectories with vLLM."""
    from vllm import SamplingParams
    
    # Tokenize all texts
    all_input_ids = []
    for text, _, _ in trajectory_items:
        input_ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        all_input_ids.append(input_ids)
    
    # Use vLLM's prompt_logprobs feature to get logits for all positions
    # Note: This uses generate with max_new_tokens=0 to get logprobs only
    sampling_params = SamplingParams(
        max_tokens=1,  # Generate 1 token to get logprobs
        prompt_logprobs=top_k,  # Get top-K logprobs for all prompt tokens
        logprobs=top_k,  # Get top-k logprobs
    )
    
    # Generate to get logprobs
    outputs = llm.generate(
        prompt_token_ids=all_input_ids,
        sampling_params=sampling_params,
    )
    
    # Extract and save logprobs
    for output, idx, input_ids, traj_item in zip(outputs, indices, all_input_ids, trajectory_items):
        output_path = output_dir / f"trajectory_{idx:06d}.npz"
        text, assistant_ranges, prompt_boundary_char = traj_item
        
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
            valid_mask = np.ones(seq_len, dtype=np.int32)  # Track positions with real logprobs
            
            for pos, logprob_dict in enumerate(prompt_logprobs):
                if logprob_dict is None:
                    valid_mask[pos] = 0  # Mark as invalid (no logprobs from vLLM)
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
            
            # Build loss mask and prompt boundary
            loss_mask, prompt_boundary = _build_loss_mask(
                text, assistant_ranges, prompt_boundary_char,
                tokenizer, input_ids, max_tokens,
            )
            
            # AND valid_mask into loss_mask: zero-filled positions never contribute to loss
            loss_mask = loss_mask * valid_mask[:len(loss_mask)]
            
            # Save with format marker, loss mask, and prompt boundary
            np.savez_compressed(
                output_path,
                values=values,
                indices=token_indices,
                input_ids=np.array(input_ids, dtype=np.int32),
                format=np.array(['logprobs']),  # Mark as log-probs from vLLM
                loss_mask=loss_mask,
                prompt_boundary=np.array([prompt_boundary], dtype=np.int32),
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
        
        trajectory_text, _, _ = build_trajectory_text(row["trajectory"])
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
            # Mock loss_mask: first 128 tokens = prompt (mask=0), rest = assistant (mask=1)
            mock_loss_mask = np.zeros(seq_len, dtype=np.int32)
            mock_loss_mask[128:] = 1
            np.savez_compressed(
                output_path,
                values=np.random.randn(seq_len, args.top_k).astype(np.float16),
                indices=np.random.randint(0, 150000, (seq_len, args.top_k), dtype=np.int32),
                input_ids=np.random.randint(0, 150000, (1, seq_len), dtype=np.int32),
                loss_mask=mock_loss_mask,
                prompt_boundary=np.array([128], dtype=np.int32),
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
