# Hypernetwork-Based Knowledge Distillation (Hyper-Agent)

Distill **Qwen3-Coder-480B** into **Qwen3-Coder-Next** (80B MoE) using a Hypernetwork that generates dynamic LoRA adapters. Optimized for full 131K-token trajectory distillation on 8×H200 GPUs.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pre-flight tests (CRITICAL - do this first!)
python tests/test_gradient_flow.py

# 3. Cache teacher logits (10-20x faster than runtime inference)
python scripts/cache_teacher_vllm.py --tp 8 --max_trajectories 30000

# 4. Start training
python scripts/train.py --config configs/train_config.yaml

# Multi-GPU training
torchrun --nproc_per_node=8 scripts/train.py --config configs/train_config.yaml
```

## Architecture

```
┌─────────────────┐     Prompt      ┌─────────────────────────────────┐
│                 │   Embeddings    │         Hypernetwork            │
│     Student     │ ────────────────│                                 │
│  (Qwen3-Next)   │                 │  ┌─────────────────────────┐  │
│                 │                 │  │    PromptEncoder        │  │
│    (frozen)     │     LoRA        │  │    (Attention Pooling)  │  │
│                 │ ◄───────────────│  └───────────┬─────────────┘  │
└────────┬────────┘    Weights      │              │                │
         │                          │  ┌───────────▼─────────────┐  │
         │                          │  │ ShapeGroupedLoRAGenerator│  │
         │                          │  │ (Per-Shape Output Heads) │  │
         │                          │  └─────────────────────────┘  │
         │                          └─────────────────────────────────┘
         │ Forward Pass (with Batched LoRA Hooks)
         ▼
    Student Logits
         │
         │ Chunked KL Divergence (2.4GB/chunk vs 318GB full)
         ▼
    Cached Teacher Logits (.npz, Top-128)
```

## Key Features

- **131K Context Distillation**: Full SWE-bench trajectories without truncation
- **Chunked KL Loss**: Avoids 318GB logit tensors by processing in 1024-token chunks
- **Batched LoRA Hooks**: Per-sample LoRA in a single forward pass via `torch.einsum`
- **Offline Teacher Caching**: vLLM-based logit pre-computation for 10-20x speedup
- **Shape-Grouped Generation**: Efficient LoRA generation with per-(in_features, out_features) heads

## Core Components

### Hypernetwork (`src/hypernetwork.py`)

| Component | Description |
|-----------|-------------|
| `PromptEncoder` | Attention-pooling to compress prompt → fixed context vector |
| `ShapeGroupedLoRAGenerator` | Generates LoRA A/B matrices grouped by layer shape |
| `AgenticHyperNetwork` | Main orchestrator combining encoder + generator |

**Targets All Projection Layers**: Both DeltaNet (75%) and Attention (25%) layers use identical projection names (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

### LoRA Injection (`src/lora_injection.py`)

| Component | Description |
|-----------|-------------|
| `HookBasedLoRAInjector` | Context manager for clean hook-based LoRA injection |
| `make_batched_lora_hook` | Per-sample LoRA via einsum for batched training |
| `discover_target_layers` | Auto-discovers target layers with dimensions |
| `LayerInfo` | Named tuple: `(name, in_features, out_features)` |

### Training (`src/training.py`)

| Component | Description |
|-----------|-------------|
| `DistillationTrainer` | Main trainer class with setup, checkpointing, logging |
| `compute_distillation_loss` | Chunked KL divergence with attention masking |
| `train_step_parallel` | Batched training step with per-sample LoRA |
| `_compute_batched_sparse_kl_loss` | Efficient sparse KL from cached top-k logits |

### Data Pipeline (`src/data_loader.py`)

| Component | Description |
|-----------|-------------|
| `CachedDistillationDataset` | Loads pre-computed teacher logits from `.npz` files |
| `SWEBenchTrajectoryDataset` | Direct HuggingFace loading (testing/runtime) |
| `collate_fn_with_teacher` | Handles variable-length sequences with padding |

### Cluster Configuration (`src/cluster_config.py`)

Optimized for **8×H200** (1128GB VRAM):

- **Student + Hypernetwork**: All 8 GPUs (offline distillation mode)
- **Teacher Caching**: vLLM with TP=8 for pre-computation
- **VRAM Budget**: ~160GB student + ~10GB hypernetwork + ~800GB activations

## Dataset

**Dataset**: `nebius/SWE-rebench-openhands-trajectories`

| Metric | Value |
|--------|-------|
| Total Trajectories | 67,074 |
| Resolved (Gold) | ~32,000 |
| Max Context | 131,072 tokens |
| Source | Qwen3-Coder-480B + OpenHands |

## Training Configuration

Default config (`configs/train_config.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 8 | Max safe at 131K context |
| Learning Rate | 1e-4 | With 1e-5 warmup |
| Epochs | 20 | Plus 2 warmup epochs |
| LoRA Rank | 16 | Alpha = 32 |
| Top-K Logits | 128 | For sparse KL |
| Max Trajectory | 131,072 | Full context |

**Estimated Runtime**: 3-5 days on 8×H200

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Main training entry point |
| `scripts/cache_teacher_vllm.py` | Pre-compute teacher logits with vLLM |
| `scripts/cache_teacher.py` | Transformers fallback for caching |
| `scripts/download_data.py` | Download dataset from HuggingFace |

## Pre-flight Verification

**CRITICAL**: Always run before training:

```bash
python tests/test_gradient_flow.py
```

Validates:
- ✓ Gradients reach the Hypernetwork from KL loss
- ✓ Zero-init preserves Student behavior at t=0
- ✓ Gradient magnitudes are healthy (no explosions/vanishing)

## Project Structure

```
hypernetwork/
├── configs/
│   └── train_config.yaml      # Training configuration
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   └── WALKTHROUGH.md
├── scripts/
│   ├── train.py               # Training entry point
│   ├── cache_teacher_vllm.py  # vLLM-based teacher caching
│   ├── cache_teacher.py       # Transformers fallback
│   └── download_data.py       # Dataset download
├── src/
│   ├── hypernetwork.py        # Hypernetwork architecture
│   ├── lora_injection.py      # Hook-based LoRA injection
│   ├── training.py            # Training loop & loss
│   ├── data_loader.py         # Dataset & caching
│   └── cluster_config.py      # 8×H200 cluster config
├── tests/
│   └── test_gradient_flow.py  # Pre-flight validation
├── requirements.txt
└── README.md
```

## Requirements

- Python ≥3.10
- PyTorch ≥2.4.0
- Transformers ≥4.51.0
- vLLM ≥0.6.0 (for teacher caching)
- Flash Attention ≥2.7.0
- 8×H200 GPUs (1128GB total VRAM)

## License

MIT
