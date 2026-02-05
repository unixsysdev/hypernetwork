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
    
    NOTE: Currently the training loop uses device_map="auto" (Accelerate-style)
    for the frozen student since it only does forward passes. FSDP wrapping is
    only needed if you want gradient-sharded training of the student weights.
    This function is provided for future use when unfreezing student layers.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial
    
    # Set up mixed precision policy
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap at the decoder layer level for proper sharding granularity.
    # The actual layer class must match the model architecture.
    # For Qwen3 models, this is typically the decoder layer class.
    # TODO: Detect the correct layer class from the model automatically.
    wrap_policy = None
    for name, module in model.named_modules():
        # Heuristic: find the repeated decoder layer class
        if hasattr(module, "self_attn") and hasattr(module, "mlp"):
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={type(module)},
            )
            break
    
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for gradient checkpointing
        auto_wrap_policy=wrap_policy,
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




if __name__ == "__main__":
    # Print cluster configuration
    config = ClusterConfig()
    config.print_summary()
    
    if config.validate():
        print("\n✓ VRAM budget validated - configuration is feasible")
    else:
        print("\n✗ VRAM budget exceeded - reduce batch size or context length")
