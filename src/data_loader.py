"""
Data Loader for SWE-bench Trajectories with Cached Teacher Logits.

Supports two modes:
1. Standard: Downloads trajectories from HuggingFace Hub
2. Cached: Loads pre-computed teacher logits from .npz files

The cached mode is REQUIRED for training efficiency since loading
the 480B teacher at runtime is impractical.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CachedDistillationDataset(Dataset):
    """
    Dataset that loads both trajectories AND pre-computed teacher logits.
    
    This is the PRODUCTION dataset for actual training. It loads:
    - Tokenized trajectory (input_ids, attention_mask)
    - Pre-computed teacher top-k logits (values, indices)
    
    The teacher logits come from the cache_teacher.py script.
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_prompt_tokens: int = 512,
        max_trajectory_tokens: int = 8192,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Args:
            cache_dir: Directory containing trajectory_NNNNNN.npz files
            max_prompt_tokens: Max tokens for Hypernetwork input
            max_trajectory_tokens: Max tokens for full trajectory
            tokenizer: Tokenizer (for pad_token_id)
        """
        self.cache_dir = Path(cache_dir)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_trajectory_tokens = max_trajectory_tokens
        
        self.pad_token_id = 0
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        
        # Find all cached files
        self.cache_files = sorted(self.cache_dir.glob("trajectory_*.npz"))
        logger.info(f"Found {len(self.cache_files)} cached trajectories in {cache_dir}")
        
        if len(self.cache_files) == 0:
            raise ValueError(f"No cached trajectories found in {cache_dir}. "
                           "Run scripts/cache_teacher.py first.")
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single cached trajectory with teacher logits."""
        cache_path = self.cache_files[idx]
        
        # Load the .npz file
        data = np.load(cache_path)
        
        # Input IDs from cache
        input_ids = torch.from_numpy(data["input_ids"].astype(np.int64)).squeeze(0)
        
        # Truncate if needed
        if len(input_ids) > self.max_trajectory_tokens:
            input_ids = input_ids[:self.max_trajectory_tokens]
        
        # Create attention mask (1 for real tokens)
        attention_mask = torch.ones_like(input_ids)
        
        # Extract prompt tokens for Hypernetwork
        prompt_ids = input_ids[:self.max_prompt_tokens]
        prompt_mask = attention_mask[:self.max_prompt_tokens]
        
        # Pad prompt if shorter
        if len(prompt_ids) < self.max_prompt_tokens:
            pad_len = self.max_prompt_tokens - len(prompt_ids)
            prompt_ids = torch.cat([
                prompt_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])
            prompt_mask = torch.cat([
                prompt_mask,
                torch.zeros(pad_len, dtype=torch.long)
            ])
        
        # Load teacher logits (top-k only)
        teacher_values = torch.from_numpy(data["values"].astype(np.float32))
        teacher_indices = torch.from_numpy(data["indices"].astype(np.int64))
        
        # Truncate teacher logits to match trajectory length
        seq_len = len(input_ids)
        if teacher_values.shape[0] > seq_len:
            teacher_values = teacher_values[:seq_len]
            teacher_indices = teacher_indices[:seq_len]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "teacher_values": teacher_values,      # [seq_len, top_k]
            "teacher_indices": teacher_indices,    # [seq_len, top_k]
        }


class SWEBenchTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for SWE-bench trajectories (without cached teacher logits).
    
    Use this for:
    - Testing the pipeline
    - When running teacher inference at runtime (slow, requires massive VRAM)
    
    For production training, use CachedDistillationDataset instead.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_prompt_tokens: int = 512,
        max_trajectory_tokens: int = 8192,
        cache_dir: Optional[str] = None,
        filter_resolved: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_trajectory_tokens = max_trajectory_tokens
        self.cache_dir = cache_dir
        self.filter_resolved = filter_resolved
        
        self.samples: List[Dict[str, Any]] = []
        
    def load_from_hub(self) -> int:
        """Load dataset from HuggingFace Hub."""
        logger.info("Loading nebius/SWE-rebench-openhands-trajectories...")
        
        ds = load_dataset(
            "nebius/SWE-rebench-openhands-trajectories",
            split="train",
            cache_dir=self.cache_dir,
        )
        
        original_count = len(ds)
        logger.info(f"Loaded {original_count} total trajectories")
        
        if self.filter_resolved:
            ds = ds.filter(
                lambda x: x["resolved"] == 1 and x["exit_status"] == "submit",
                num_proc=4,
            )
            logger.info(f"Filtered to {len(ds)} resolved trajectories")
        
        for row in ds:
            processed = self._process_trajectory(row)
            if processed is not None:
                self.samples.append(processed)
        
        logger.info(f"Processed {len(self.samples)} valid samples")
        return len(self.samples)
    
    def _process_trajectory(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single trajectory into tokenized format."""
        trajectory = row["trajectory"]
        
        if not trajectory:
            return None
        
        # Build messages for chat template
        messages = []
        for msg in trajectory:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Handle tool calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                tool_content = []
                for tc in tool_calls:
                    if tc.get("function"):
                        func = tc["function"]
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                pass
                        tool_content.append(f"<tool_call>\n{name}({json.dumps(args)})\n</tool_call>")
                content = content + "\n".join(tool_content) if content else "\n".join(tool_content)
            
            # Handle tool responses - check if tokenizer supports "tool" role
            if role == "tool":
                # Try to keep as "tool" if the tokenizer supports it
                # Otherwise fall back to wrapping in user message
                tool_name = msg.get("name", "tool")
                content = f"<tool_response name=\"{tool_name}\">\n{content}\n</tool_response>"
                # Check if tokenizer chat template supports tool role
                try:
                    # Test if tool role works
                    self.tokenizer.apply_chat_template(
                        [{"role": "tool", "content": "test"}],
                        tokenize=False,
                    )
                except Exception:
                    # Tokenizer doesn't support tool role, map to user
                    role = "user"
            
            if content:
                messages.append({"role": role, "content": content})
        
        if not messages:
            return None
        
        # Apply chat template
        try:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")
            full_text = "\n\n".join([f"[{m['role']}]: {m['content']}" for m in messages])
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_trajectory_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        
        if len(input_ids) < 100:
            return None
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "instance_id": row.get("instance_id", ""),
            "repo": row.get("repo", ""),
            "trajectory_id": row.get("trajectory_id", ""),
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
        
        prompt_ids = input_ids[:self.max_prompt_tokens]
        prompt_mask = attention_mask[:self.max_prompt_tokens]
        
        if len(prompt_ids) < self.max_prompt_tokens:
            pad_len = self.max_prompt_tokens - len(prompt_ids)
            prompt_ids = torch.cat([
                prompt_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long)
            ])
            prompt_mask = torch.cat([
                prompt_mask,
                torch.zeros(pad_len, dtype=torch.long)
            ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
        }
    
    def save_to_arrow(self, path: str) -> None:
        """Save to Arrow format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "input_ids": [s["input_ids"] for s in self.samples],
            "attention_mask": [s["attention_mask"] for s in self.samples],
            "instance_id": [s["instance_id"] for s in self.samples],
            "repo": [s["repo"] for s in self.samples],
            "trajectory_id": [s["trajectory_id"] for s in self.samples],
        }
        
        table = pa.table(data)
        pq.write_table(table, str(path))
        logger.info(f"Saved {len(self.samples)} samples to {path}")
    
    def load_from_arrow(self, path: str) -> int:
        """Load from Arrow format."""
        table = pq.read_table(path)
        df = table.to_pandas()
        
        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                "input_ids": list(row["input_ids"]),
                "attention_mask": list(row["attention_mask"]),
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "trajectory_id": row["trajectory_id"],
            })
        
        logger.info(f"Loaded {len(self.samples)} samples from {path}")
        return len(self.samples)


def collate_fn_with_teacher(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles teacher logits.
    """
    batch_max_len = max(len(item["input_ids"]) for item in batch)
    if max_length:
        batch_max_len = min(batch_max_len, max_length)
    
    has_teacher = "teacher_values" in batch[0]
    top_k = batch[0]["teacher_values"].shape[-1] if has_teacher else 128
    
    input_ids_list = []
    attention_mask_list = []
    prompt_ids_list = []
    prompt_mask_list = []
    teacher_values_list = []
    teacher_indices_list = []
    
    for item in batch:
        ids = item["input_ids"][:batch_max_len]
        mask = item["attention_mask"][:batch_max_len]
        
        pad_len = batch_max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        prompt_ids_list.append(item["prompt_ids"])
        prompt_mask_list.append(item["prompt_mask"])
        
        if has_teacher:
            tv = item["teacher_values"][:batch_max_len]
            ti = item["teacher_indices"][:batch_max_len]
            
            if tv.shape[0] < batch_max_len:
                pad_len_t = batch_max_len - tv.shape[0]
                tv = torch.cat([tv, torch.zeros((pad_len_t, top_k))])
                ti = torch.cat([ti, torch.zeros((pad_len_t, top_k), dtype=torch.long)])
            
            teacher_values_list.append(tv)
            teacher_indices_list.append(ti)
    
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "prompt_ids": torch.stack(prompt_ids_list),
        "prompt_mask": torch.stack(prompt_mask_list),
    }
    
    if has_teacher:
        result["teacher_values"] = torch.stack(teacher_values_list)
        result["teacher_indices"] = torch.stack(teacher_indices_list)
    
    return result


def create_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_prompt_tokens: int = 512,
    max_trajectory_tokens: int = 8192,
    cache_dir: Optional[str] = None,
    arrow_path: Optional[str] = None,
    teacher_cache_dir: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_prompt_tokens: Max tokens for Hypernetwork input
        max_trajectory_tokens: Max tokens for trajectory
        cache_dir: HuggingFace cache dir
        arrow_path: Path to pre-tokenized Arrow file
        teacher_cache_dir: Path to cached teacher logits (from cache_teacher.py)
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader ready for training
    """
    # If teacher cache exists, use it (PRODUCTION MODE)
    if teacher_cache_dir and Path(teacher_cache_dir).exists():
        logger.info(f"Using cached teacher logits from {teacher_cache_dir}")
        dataset = CachedDistillationDataset(
            cache_dir=teacher_cache_dir,
            max_prompt_tokens=max_prompt_tokens,
            max_trajectory_tokens=max_trajectory_tokens,
            tokenizer=tokenizer,
        )
    else:
        # Fall back to loading trajectories only (TESTING MODE)
        logger.warning("No teacher cache found - Teacher will be called at runtime!")
        dataset = SWEBenchTrajectoryDataset(
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            max_trajectory_tokens=max_trajectory_tokens,
            cache_dir=cache_dir,
        )
        
        if arrow_path and Path(arrow_path).exists():
            dataset.load_from_arrow(arrow_path)
        else:
            dataset.load_from_hub()
            if arrow_path:
                dataset.save_to_arrow(arrow_path)
    
    pad_token_id = tokenizer.pad_token_id or 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_with_teacher(batch, pad_token_id, max_trajectory_tokens),
        pin_memory=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next", trust_remote_code=True)
    
    # Test cached dataset if available
    cache_dir = Path("./teacher_cache")
    if cache_dir.exists() and list(cache_dir.glob("*.npz")):
        print("Testing CachedDistillationDataset...")
        dataset = CachedDistillationDataset(
            cache_dir=str(cache_dir),
            tokenizer=tokenizer,
        )
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Teacher values shape: {sample['teacher_values'].shape}")
    else:
        print("No teacher cache found. Run cache_teacher.py first.")
