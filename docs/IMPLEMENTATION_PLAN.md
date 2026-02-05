# Hypernetwork-Based Agentic Knowledge Distillation

**Goal**: Train a Hypernetwork to dynamically generate LoRA adapters for Qwen3-Coder-Next (80B, 3B active) that enable it to match the agentic coding behavior of Qwen3-Coder-480B on SWE-bench tasks.

---

## User Review Required

> [!IMPORTANT]
> **Critical Architecture Discovery**: The Student model (Qwen3-Coder-Next) uses a **Hybrid Layout** with **DeltaNet Linear Attention** (75% of layers) + **Standard Attention** (25% of layers). If we only target standard attention layers, we lose 75% of steering capability.

> [!WARNING]
> **Hidden Dimension is Small**: The Student has `hidden_dim=2048`, which is unusually small for an 80B model. This means LoRA parameters are actually very manageable (~21-84M), but the signal is highly compressed—LoRA precision matters more.

---

## Model Specifications

### Teacher: Qwen3-Coder-480B-A35B-Instruct
| Property | Value |
|----------|-------|
| Total Params | 480B |
| Active Params | 35B |
| Layers | 62 |
| Experts / Active | 160 / 8 |
| Context Length | 262K native |
| Quantization | FP8 recommended on H200 |

### Student: Qwen3-Coder-Next (80B-3B)
| Property | Value |
|----------|-------|
| Total Params | 80B |
| Active Params | 3B |
| **Hidden Dim** | **2048** |
| Layers | 48 |
| Experts / Active | 512 / 10 |
| **Hybrid Layout** | `12 * (3 * DeltaNet + 1 * Attention)` |
| Context Length | 262K native |

### Dataset: nebius/SWE-rebench-openhands-trajectories
| Property | Value |
|----------|-------|
| Total Rows | 67,074 |
| Size | 2.08 GB |
| Generator | Qwen3-Coder-480B |
| Scaffolding | OpenHands v0.54.0 |
| Gold Filter | `resolved == 1` (~32k samples) |

---

## Proposed Changes

### Data Pipeline

#### [NEW] [data_loader.py](file:///home/marcel/Work/hypernetowrk/src/data_loader.py)
Dataset loading, filtering, and pre-tokenization pipeline.

```python
# Core functionality:
# 1. Load nebius/SWE-rebench-openhands-trajectories
# 2. Filter: resolved == 1 AND exit_status == "submit"
# 3. Extract: prompt (first 512 tokens) + trajectory (full)
# 4. Pre-tokenize and save to Arrow format
```

**Key Fields from Dataset**:
- `trajectory`: List of messages (system, assistant, user, tool)
- `resolved`: 1 = success, 0 = failure
- `exit_status`: "submit" = completed
- `model_patch`: The actual code fix

---

### Hypernetwork Architecture

#### [NEW] [hypernetwork.py](file:///home/marcel/Work/hypernetowrk/src/hypernetwork.py)
The core Hypernetwork that generates LoRA adapters.

```python
class AgenticHyperNetwork(nn.Module):
    """
    Input: Prompt embedding from Student (first 512 tokens)
    Output: LoRA weights for ALL target layers
    
    Architecture:
    - Embed pooler: Reduce [B, 512, 2048] -> [B, 2048]
    - Transformer encoder: 4 layers, 8 heads (context understanding)
    - LoRA generators: Separate MLPs per layer type
    """
    
    # Target modules for Student (CRITICAL - both architectures):
    ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 25% of layers (12 total)
    DELTANET_TARGETS = ["linear_q", "linear_k", "linear_v"]      # 75% of layers (36 total)
    
    # LoRA config
    RANK = 16
    ALPHA = 32
```

**CRITICAL Design Decisions**:

1. **Zero Initialization**: Output layer initialized to zeros/near-zeros to ensure:
   ```
   At Step 0: Student(x, LoRA) ≈ Student(x)  # No harm at start
   ```

2. **Shared Generator with Layer Embeddings**:
   - Don't generate 48 separate LoRAs
   - Use layer embedding + shared generator
   - Reduces Hypernetwork params from ~500M to ~50M

3. **Capacity Check**:
   ```
   Per-layer LoRA: 2 * hidden_dim * rank = 2 * 2048 * 16 = 65,536 params
   
   Attention layers: 12 blocks * 4 modules * 65,536 = 3.1M params
   DeltaNet layers: 36 blocks * 3 modules * 65,536 = 7.1M params
   
   Total LoRA output: ~10M params per forward pass
   Hypernetwork params: ~50M (encoder + generators)
   ```

---

### Training Framework

#### [NEW] [training.py](file:///home/marcel/Work/hypernetowrk/src/training.py)
Main training loop with Teacher Forcing.

```python
def train_step(batch):
    """
    1. Hypernetwork generates LoRA from PROMPT ONLY
    2. Student forward pass with LoRA (full trajectory)
    3. Teacher forward pass (full trajectory) - just scoring
    4. KL Divergence loss on logits
    5. Backprop to Hypernetwork only
    """
    
    # Extract prompt (first 512 tokens) and full trajectory
    prompt_ids = batch["input_ids"][:, :512]
    full_ids = batch["input_ids"]
    
    # 1. Hypernetwork generates LoRA
    with torch.no_grad():
        prompt_embed = student.get_input_embeddings()(prompt_ids)
    lora_weights = hypernetwork(prompt_embed)  # Only place gradients flow
    
    # 2. Student forward (with LoRA injection)
    student_logits = student_forward_with_lora(full_ids, lora_weights)
    
    # 3. Teacher forward (scoring only - no generation!)
    with torch.no_grad():
        teacher_logits = teacher(full_ids).logits
    
    # 4. KL Divergence (Top-K for bandwidth efficiency)
    loss = top_k_kl_divergence(student_logits, teacher_logits, k=128)
    
    # 5. Backprop (gradients flow: loss -> student -> lora_weights -> hypernetwork)
    loss.backward()
    optimizer.step()
```

**Speed Optimization (Critical)**:
- **No Generation**: Both Teacher and Student do **forward pass only** (Teacher Forcing)
- **Parallel Processing**: All tokens processed simultaneously
- **Top-K Logits**: Transfer only top-128 logits (vocab 150k -> 128 = 1000x reduction)

#### [NEW] [lora_injection.py](file:///home/marcel/Work/hypernetowrk/src/lora_injection.py)
Dynamic LoRA weight injection into frozen Student.

```python
def student_forward_with_lora(input_ids, lora_weights):
    """
    Inject LoRA weights into Student's attention layers.
    
    CRITICAL: Student is FROZEN. Only LoRA weights change.
    The gradient flows through the LoRA weights back to Hypernetwork.
    """
    # Decompose lora_weights into per-layer A, B matrices
    # For each target layer:
    #   output = base_layer(x) + (x @ A) @ B * (alpha / rank)
```

---

### Infrastructure Setup

#### [NEW] [cluster_config.py](file:///home/marcel/Work/hypernetowrk/src/cluster_config.py)
8x H200 configuration with segregated GPU groups.

```python
"""
VRAM Budget (8 x 141GB = 1,128 GB total):

Teacher (FP8): ~280 GB -> GPUs 0-3 (vLLM, TP=4)
Student (BF16): ~160 GB -> GPUs 4-7 (Training)
Hypernetwork: ~10 GB -> GPUs 4-7
Activations: ~200 GB (8k context, batch 8)

TOTAL: ~650 GB used, ~480 GB headroom for scaling
"""

# Teacher cluster (Inference)
TEACHER_CONFIG = {
    "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "quantization": "fp8",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.90,
    "devices": [0, 1, 2, 3],
}

# Student cluster (Training)
STUDENT_CONFIG = {
    "model": "Qwen/Qwen3-Coder-Next",
    "dtype": "bfloat16",
    "devices": [4, 5, 6, 7],
    "fsdp": True,  # Fully Sharded Data Parallel
}
```

---

### Verification Tests

#### [NEW] [tests/test_gradient_flow.py](file:///home/marcel/Work/hypernetowrk/tests/test_gradient_flow.py)
Unit test to verify gradients flow from loss to Hypernetwork.

```python
def test_gradient_flow():
    """
    CRITICAL: Run before full training.
    Verifies the gradient highway is intact.
    """
    # 1. Freeze Student, unfreeze Hypernetwork
    # 2. Forward pass
    # 3. Dummy loss
    # 4. Backward pass
    # 5. Check: hypernetwork.output_layer.weight.grad != None
    # 6. Check: grad magnitude > 0
```

---

## Training Schedule

| Phase | Duration | Description |
|-------|----------|-------------|
| **Warmup** | 4 hours | 1k samples, LR=1e-5, validate loss decreases |
| **Main Training** | 32 hours | Full 20k samples, 20 epochs, LR=1e-4 |
| **Context Annealing** | 8 hours | Extend to 32k context for long trajectories |

### Training Math

```
Trajectories: 20,000 (filtered gold set)
Avg Sequence Length: 8,000 tokens
Total Tokens/Epoch: 160M tokens

Cluster Speed: 8x H200 @ 3,500 tok/s = 28,000 tok/s
Time/Epoch: 160M / 28,000 = 5,714s ≈ 1.6 hours

Total (20 epochs): 1.6h * 20 = 32 hours
```

---

## Verification Plan

### Automated Tests
1. **Gradient Flow Test**: Verify backprop reaches Hypernetwork
2. **Zero-Init Test**: Confirm Student(x, LoRA_init) ≈ Student(x)
3. **Loss Convergence**: Monitor KL divergence decreases over epochs

### Manual Verification
1. **Qualitative Check**: Run Student+LoRA on held-out SWE-bench problems
2. **Compare**: Student+LoRA vs Base Student on same problems
3. **Tool Use**: Verify model correctly predicts tool calls

---

## Project Structure

```
hypernetowrk/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── hypernetwork.py     # Hypernetwork architecture
│   ├── lora_injection.py   # Dynamic LoRA injection
│   ├── training.py         # Training loop
│   ├── cluster_config.py   # Infrastructure setup
│   └── utils.py            # Logging, checkpointing
├── tests/
│   ├── test_gradient_flow.py
│   └── test_zero_init.py
├── scripts/
│   ├── download_data.py    # Download and filter dataset
│   ├── pretokenize.py      # Pre-tokenization to Arrow
│   └── train.py            # Main training entry point
├── configs/
│   └── train_config.yaml   # Hyperparameters
└── requirements.txt
```

---

## Dependencies

```txt
torch>=2.4.0
transformers>=4.51.0
datasets>=3.0.0
peft>=0.14.0
vllm>=0.6.0
flash-attn>=2.7.0
accelerate>=1.0.0
wandb
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Gradient explosion at start | Zero-init Hypernetwork output layer |
| DeltaNet layers not steered | Explicitly target `linear_q/k/v` in addition to `q_proj` |
| VRAM overflow | Use Top-K logits, gradient checkpointing |
| Tokenizer mismatch | Verified: same Qwen3 tokenizer for both models |
| Data quality | Filter `resolved=1` only (verified trajectories) |
