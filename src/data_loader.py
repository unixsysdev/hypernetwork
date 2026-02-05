"""
Data Loader for SWE-bench Trajectories.

Downloads, filters, and pre-tokenizes the nebius/SWE-rebench-openhands-trajectories
dataset for training the Hypernetwork.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class SWEBenchTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for SWE-bench trajectories.
    
    Each sample contains:
    - prompt_ids: First N tokens (for Hypernetwork input)
    - input_ids: Full trajectory tokens (for Teacher Forcing)
    - attention_mask: Attention mask for the full trajectory
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
        """
        Load dataset from HuggingFace Hub and filter for successful trajectories.
        
        Returns:
            Number of samples loaded
        """
        logger.info("Loading nebius/SWE-rebench-openhands-trajectories from Hub...")
        
        ds = load_dataset(
            "nebius/SWE-rebench-openhands-trajectories",
            split="train",
            cache_dir=self.cache_dir,
        )
        
        original_count = len(ds)
        logger.info(f"Loaded {original_count} total trajectories")
        
        if self.filter_resolved:
            # Filter for successful trajectories only
            ds = ds.filter(
                lambda x: x["resolved"] == 1 and x["exit_status"] == "submit",
                num_proc=4,
            )
            logger.info(f"Filtered to {len(ds)} resolved trajectories (Gold Set)")
        
        # Process trajectories
        for row in ds:
            processed = self._process_trajectory(row)
            if processed is not None:
                self.samples.append(processed)
        
        logger.info(f"Processed {len(self.samples)} valid samples")
        return len(self.samples)
    
    def _process_trajectory(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single trajectory into tokenized format.
        
        Extracts the prompt (first message) and full trajectory text.
        """
        trajectory = row["trajectory"]
        
        if not trajectory:
            return None
        
        # Build the full conversation text using chat template
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
            
            # Handle tool responses
            if role == "tool":
                role = "user"  # Map tool to user for tokenizer compatibility
                tool_name = msg.get("name", "tool")
                content = f"<tool_response name=\"{tool_name}\">\n{content}\n</tool_response>"
            
            if content:
                messages.append({"role": role, "content": content})
        
        if not messages:
            return None
        
        # Apply chat template to get the full text
        try:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback: concatenate messages manually
            full_text = "\n\n".join([f"[{m['role']}]: {m['content']}" for m in messages])
        
        # Tokenize the full trajectory
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_trajectory_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        
        if len(input_ids) < 100:  # Skip very short trajectories
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
        
        # Extract prompt tokens (first N tokens for Hypernetwork)
        prompt_ids = input_ids[:self.max_prompt_tokens]
        prompt_mask = attention_mask[:self.max_prompt_tokens]
        
        # Pad prompt if shorter than max_prompt_tokens
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
        """Save processed samples to Arrow format for fast loading."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to Arrow table
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
        """Load pre-tokenized samples from Arrow format."""
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


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within the batch.
    """
    # Find max length in batch
    batch_max_len = max(len(item["input_ids"]) for item in batch)
    if max_length:
        batch_max_len = min(batch_max_len, max_length)
    
    input_ids_list = []
    attention_mask_list = []
    prompt_ids_list = []
    prompt_mask_list = []
    
    for item in batch:
        ids = item["input_ids"][:batch_max_len]
        mask = item["attention_mask"][:batch_max_len]
        
        # Pad to batch_max_len
        pad_len = batch_max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        prompt_ids_list.append(item["prompt_ids"])
        prompt_mask_list.append(item["prompt_mask"])
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "prompt_ids": torch.stack(prompt_ids_list),
        "prompt_mask": torch.stack(prompt_mask_list),
    }


def create_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_prompt_tokens: int = 512,
    max_trajectory_tokens: int = 8192,
    cache_dir: Optional[str] = None,
    arrow_path: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    If arrow_path exists, loads from there. Otherwise downloads and processes.
    """
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
        collate_fn=lambda batch: collate_fn(batch, pad_token_id, max_trajectory_tokens),
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next")
    
    dataset = SWEBenchTrajectoryDataset(
        tokenizer=tokenizer,
        max_prompt_tokens=512,
        max_trajectory_tokens=8192,
    )
    
    # Load a small sample for testing
    count = dataset.load_from_hub()
    print(f"Loaded {count} samples")
    
    if count > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Prompt IDs shape: {sample['prompt_ids'].shape}")
