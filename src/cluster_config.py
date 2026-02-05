"""
Cluster Configuration for 8x H200 Training.

This module defines the infrastructure setup for distributed training:
- Teacher on GPUs 0-3 (vLLM, FP8, Tensor Parallel)
- Student + Hypernetwork on GPUs 4-7 (FSDP, BF16)

The segregated approach prevents interference between:
- Teacher inference (high throughput, no gradients)
- Student training (gradient computation, optimizer states)
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.distributed as dist


@dataclass
class TeacherConfig:
    """Configuration for the Teacher model (inference only)."""
    model_id: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    quantization: str = "fp8"  # fp8, int8, or none
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.90
    devices: List[int] = None
    max_model_len: int = 8192  # Context length for inference
    
    def __post_init__(self):
        if self.devices is None:
            self.devices = [0, 1, 2, 3]


@dataclass 
class StudentConfig:
    """Configuration for the Student model (training)."""
    model_id: str = "Qwen/Qwen3-Coder-Next"
    dtype: str = "bfloat16"
    devices: List[int] = None
    use_fsdp: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        if self.devices is None:
            self.devices = [4, 5, 6, 7]


@dataclass
class ClusterConfig:
    """Complete cluster configuration."""
    teacher: TeacherConfig = None
    student: StudentConfig = None
    
    # VRAM budget (in GB)
    total_vram: float = 1128.0  # 8 x 141GB H200
    teacher_vram: float = 280.0  # FP8 480B
    student_vram: float = 160.0  # BF16 80B
    hypernetwork_vram: float = 10.0
    activation_headroom: float = 500.0  # For large batches
    
    def __post_init__(self):
        if self.teacher is None:
            self.teacher = TeacherConfig()
        if self.student is None:
            self.student = StudentConfig()
    
    def validate(self) -> bool:
        """Check that VRAM budget is satisfied."""
        total_used = (
            self.teacher_vram + 
            self.student_vram + 
            self.hypernetwork_vram + 
            self.activation_headroom
        )
        return total_used <= self.total_vram
    
    def print_summary(self):
        """Print cluster configuration summary."""
        print("=" * 60)
        print("CLUSTER CONFIGURATION")
        print("=" * 60)
        print(f"\nTotal VRAM: {self.total_vram:.0f} GB")
        print(f"\nTeacher: {self.teacher.model_id}")
        print(f"  - Quantization: {self.teacher.quantization}")
        print(f"  - Tensor Parallel: {self.teacher.tensor_parallel_size}")
        print(f"  - GPUs: {self.teacher.devices}")
        print(f"  - VRAM: ~{self.teacher_vram:.0f} GB")
        print(f"\nStudent: {self.student.model_id}")
        print(f"  - Dtype: {self.student.dtype}")
        print(f"  - FSDP: {self.student.use_fsdp}")
        print(f"  - GPUs: {self.student.devices}")
        print(f"  - VRAM: ~{self.student_vram:.0f} GB")
        print(f"\nHypernetwork VRAM: ~{self.hypernetwork_vram:.0f} GB")
        print(f"Activation Headroom: ~{self.activation_headroom:.0f} GB")
        print(f"\nTotal Used: {self.teacher_vram + self.student_vram + self.hypernetwork_vram + self.activation_headroom:.0f} GB")
        print(f"Headroom: {self.total_vram - (self.teacher_vram + self.student_vram + self.hypernetwork_vram + self.activation_headroom):.0f} GB")
        print("=" * 60)


def setup_vllm_teacher(config: TeacherConfig) -> Any:
    """
    Set up the Teacher model using vLLM for efficient inference.
    
    vLLM provides:
    - PagedAttention for memory efficiency
    - Continuous batching
    - Native FP8 quantization on H200
    - Tensor Parallelism for large models
    
    Returns:
        vLLM LLM object
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for Teacher inference. "
            "Install with: pip install vllm"
        )
    
    # Set visible devices for Teacher
    # WARNING: This globally modifies the environment. If running Teacher and Student
    # in the same process, use device placement or process isolation instead.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))
    
    # Initialize vLLM
    llm = LLM(
        model=config.model_id,
        quantization=config.quantization,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
    )
    
    return llm


# NOTE: get_teacher_logits_vllm was removed - vLLM logit extraction is implemented
# in scripts/cache_teacher_vllm.py using prompt_logprobs instead.


def setup_fsdp_student(
    config: StudentConfig,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Wrap the Student model with FSDP for distributed training.
    
    FSDP (Fully Sharded Data Parallel) shards:
    - Model parameters
    - Optimizer states
    - Gradients
    
    This allows training larger models than would fit on a single GPU.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
    )
    
    # Set up mixed precision policy
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Wrap with FSDP
    # Note: The actual wrapping depends on the model architecture
    # This is a template that needs to be customized for Qwen3
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for gradient checkpointing
    )
    
    return fsdp_model


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class ProducerConsumerPipeline:
    """
    Async pipeline for overlapping Teacher inference with Student training.
    
    Producer (GPUs 0-3): Continuously runs Teacher forward passes
    Consumer (GPUs 4-7): Trains on the produced logits
    
    Communication via a shared queue (CPU memory or CUDA IPC).
    """
    
    def __init__(
        self,
        queue_size: int = 32,
        use_cuda_ipc: bool = False,
        max_prompt_tokens: int = 512,
    ):
        import queue
        import threading
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.use_cuda_ipc = use_cuda_ipc
        self.max_prompt_tokens = max_prompt_tokens
        self._stop_event = threading.Event()
        
    def producer_loop(self, teacher, dataloader, top_k: int = 128):
        """
        Producer thread: Generate Teacher logits.
        
        Runs on GPUs 0-3. Converts dense logits to sparse top-k format
        to match what train_step expects.
        """
        for batch in dataloader:
            if self._stop_event.is_set():
                break
                
            with torch.no_grad():
                teacher_logits = teacher(batch["input_ids"]).logits  # [B, L, V]
                
                # Convert to sparse top-k format (matches cached teacher format)
                teacher_values, teacher_indices = torch.topk(teacher_logits, top_k, dim=-1)
                
                # Move to CPU to decouple from Teacher GPUs
                if not self.use_cuda_ipc:
                    teacher_values = teacher_values.cpu()
                    teacher_indices = teacher_indices.cpu()
            
            # Put in queue with keys matching train_step expectations
            max_len = self.max_prompt_tokens
            self.queue.put({
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "prompt_ids": batch.get("prompt_ids", batch["input_ids"][:, :max_len]),
                "prompt_mask": batch.get("prompt_mask", batch["attention_mask"][:, :max_len]),
                "teacher_values": teacher_values,    # Sparse format
                "teacher_indices": teacher_indices,  # Sparse format
                "teacher_format": "logits",          # These are raw logits, not log-probs
            })
    
    def consumer_loop(self, trainer, num_steps: int):
        """
        Consumer thread: Train on the logits.
        
        Runs on GPUs 4-7.
        """
        for _ in range(num_steps):
            if self._stop_event.is_set():
                break
                
            # Get from queue (blocks if empty)
            batch = self.queue.get()
            
            # Move Teacher logits to training device
            if not self.use_cuda_ipc:
                batch["teacher_values"] = batch["teacher_values"].cuda()
                batch["teacher_indices"] = batch["teacher_indices"].cuda()
            
            # Train step
            trainer.train_step(batch)
    
    def stop(self):
        """Signal both threads to stop."""
        self._stop_event.set()


if __name__ == "__main__":
    # Print cluster configuration
    config = ClusterConfig()
    config.print_summary()
    
    if config.validate():
        print("\n✓ VRAM budget validated - configuration is feasible")
    else:
        print("\n✗ VRAM budget exceeded - reduce batch size or context length")
