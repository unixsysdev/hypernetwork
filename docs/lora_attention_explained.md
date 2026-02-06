# LoRA and Attention: A Complete Mental Model

This document explains how attention works in transformer models, how a token's vector flows through attention and interacts with other tokens, and exactly how LoRA attaches to modify attention behavior.

---

## Part 1: How Attention Works

### The Input

Let's say we have a sequence of **5 tokens**: `["The", "cat", "sat", "on", "mat"]`

Each token has been embedded into a **hidden dimension vector** (dim=2048 for Qwen):

```
Token 0 "The": x₀ = [0.2, -0.5, 0.8, ...]  (2048 numbers)
Token 1 "cat": x₁ = [0.9, 0.1, -0.3, ...]  (2048 numbers)
Token 2 "sat": x₂ = [0.4, 0.7, 0.2, ...]   (2048 numbers)
Token 3 "on":  x₃ = [-0.1, 0.3, 0.6, ...]  (2048 numbers)
Token 4 "mat": x₄ = [0.5, -0.2, 0.4, ...]  (2048 numbers)

Input matrix X = [x₀, x₁, x₂, x₃, x₄]  shape: [5, 2048]
```

### Step 1: Create Three Views of Each Token (Q, K, V)

Each token's vector gets **projected three times** into different "roles":

```
┌─────────────────────────────────────────────────────────────────────┐
│                     For token i (e.g., "cat"):                      │
│                                                                     │
│   Query:  qᵢ = xᵢ @ W_q    "What am I looking for?"                 │
│   Key:    kᵢ = xᵢ @ W_k    "What do I contain that others seek?"    │
│   Value:  vᵢ = xᵢ @ W_v    "What information do I carry?"           │
└─────────────────────────────────────────────────────────────────────┘
```

After projection, each token now has 3 vectors:
- **Query (Q)**: Used to "ask questions" to other tokens
- **Key (K)**: Used to "answer" when other tokens ask
- **Value (V)**: The actual content to retrieve if there's a match

```
All tokens:
Q = X @ W_q  →  [q₀, q₁, q₂, q₃, q₄]   shape: [5, d_head]
K = X @ W_k  →  [k₀, k₁, k₂, k₃, k₄]   shape: [5, d_head]  
V = X @ W_v  →  [v₀, v₁, v₂, v₃, v₄]   shape: [5, d_head]
```

### Step 2: Every Token Asks "Who Should I Attend To?"

Each token's Query computes a **similarity score** with every token's Key:

```
Attention scores = Q @ K^T    (matrix multiply)
```

Expanding for token 1 ("cat"):

```
Token 1's Query: q₁

Dot product with every Key:
  score(1→0) = q₁ · k₀ = "How relevant is 'The' to 'cat'?"     → 0.3
  score(1→1) = q₁ · k₁ = "How relevant is 'cat' to itself?"   → 2.1  ← high!
  score(1→2) = q₁ · k₂ = "How relevant is 'sat' to 'cat'?"    → 1.8  ← verb relationship
  score(1→3) = q₁ · k₃ = "How relevant is 'on' to 'cat'?"     → 0.1
  score(1→4) = q₁ · k₄ = "How relevant is 'mat' to 'cat'?"    → 0.5
```

This gives us a **full attention matrix** (every token to every token):

```
              k₀     k₁     k₂     k₃     k₄
           ┌──────────────────────────────────┐
      q₀   │  2.0    0.5    0.3    0.1    0.2 │  ← "The" attends mostly to itself
      q₁   │  0.3    2.1    1.8    0.1    0.5 │  ← "cat" attends to itself + "sat"
      q₂   │  0.4    1.5    2.2    0.8    0.3 │  ← "sat" attends to "cat" + itself
      q₃   │  0.1    0.2    0.9    1.8    1.2 │  ← "on" attends to "sat" + itself + "mat"
      q₄   │  0.2    0.7    0.3    1.0    2.0 │  ← "mat" attends to "on" + itself
           └──────────────────────────────────┘
                    Raw attention scores
```

### Step 3: Normalize with Softmax (Make Weights Sum to 1)

Each row gets normalized so the weights sum to 1:

```
For token 1 ("cat"):
  Raw:    [0.3, 2.1, 1.8, 0.1, 0.5]
  After softmax: [0.05, 0.42, 0.33, 0.03, 0.07]
                   ↑      ↑     ↑
               "The"  "cat"  "sat"
```

Now token 1 has **attention weights** saying:
- Get 42% of information from "cat" (itself)
- Get 33% from "sat" (the verb it performs)
- Get 5% from "The", 3% from "on", 7% from "mat"

### Step 4: Weighted Sum of Values (The Actual Mixing)

Token 1 **retrieves information** by taking a weighted combination of everyone's Values:

```
new representation of "cat" = 
    0.05 × v₀("The") +
    0.42 × v₁("cat") +    ← get most info from self
    0.33 × v₂("sat") +    ← mix in verb information!
    0.03 × v₃("on") +
    0.07 × v₄("mat")
```

**This is where the magic happens!** The vector for "cat" now **contains information from other relevant tokens** (especially "sat" - the action it performs).

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Before attention:  x₁ = pure "cat" embedding                            │
│  After attention:   o₁ = "cat" mixed with context = "cat that sat..."   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Step 5: Project Back with O_proj

The attended output goes through one more linear layer:

```
final_output₁ = o₁ @ W_o    shape: [2048]
```

This output then goes to the MLP, residual connection, and next layer.

---

## Part 2: Visual Summary - Token Life Through Attention

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          LIFE OF TOKEN "cat" (x₁)                             │
└───────────────────────────────────────────────────────────────────────────────┘

            x₁ [2048]
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
  Q-proj     K-proj     V-proj          ← 3 linear projections
    │          │          │
    ▼          ▼          ▼
   q₁         k₁         v₁              ← 3 views of same token
   [256]     [256]      [256]

   q₁ ─────────────────────────┐
   │                           │
   │  Compare with ALL keys:   │
   │                           ▼
   │    q₁·k₀ = 0.3   ─────→  softmax  ───→  0.05   ─┐
   │    q₁·k₁ = 2.1   ─────→  softmax  ───→  0.42   ─┤
   │    q₁·k₂ = 1.8   ─────→  softmax  ───→  0.33   ─┼─→  attention weights
   │    q₁·k₃ = 0.1   ─────→  softmax  ───→  0.03   ─┤
   │    q₁·k₄ = 0.5   ─────→  softmax  ───→  0.07   ─┘
   │
   │  Weighted sum of ALL values:
   │
   │    0.05×v₀ + 0.42×v₁ + 0.33×v₂ + 0.03×v₃ + 0.07×v₄ = o₁
   │                                                       │
   │                                                       ▼
   │                                                   O-proj
   │                                                       │
   └───────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                               new_x₁ = x₁ + o₁  (residual)
                                                     [2048]

     "cat" now understands its relationship to "sat" and the whole sentence!
```

---

## Part 3: How LoRA Attaches to Attention

### The Core LoRA Idea

Instead of fine-tuning the full weight matrix `W` (which is huge), LoRA learns a **low-rank additive update**:

```
W_new = W_frozen + ΔW

Where: ΔW = A @ B  (low-rank factorization)
       A: [in_features × rank]    (e.g., [2048 × 16])
       B: [rank × out_features]   (e.g., [16 × 4096])
```

The `rank` is small (like 16), so `A` and `B` together have **far fewer parameters** than the original weight `W`.

**Key insight**: We don't modify `W` at all. We just **add** the LoRA delta to the output.

### Hook-Based Injection (The Actual Code)

From `lora_injection.py`:

```python
def make_lora_hook(lora_A, lora_B, scaling):
    def hook(module, input, output):
        x = input[0]  # [B, L, in_features] — the input to the linear layer
        
        # LoRA delta: (x @ A) @ B * scaling
        lora_delta = (x @ lora_A) @ lora_B * scaling
        
        # ADD to the base output (don't replace!)
        return output + lora_delta
    return hook
```

**What this does:**

1. **Before the hook**: The linear layer runs normally: `base_output = x @ W_frozen + bias`
2. **The hook intercepts the output**: It takes both `input` (x) and `output` (base_output)
3. **Computes the LoRA delta**: `Δ = (x @ A) @ B * scaling`
4. **Returns modified output**: `base_output + Δ`

So the final result is: `x @ W + (x @ A @ B * scaling)` = `x @ (W + A @ B * scaling)`

### Visual: LoRA Attaching to q_proj

```
                  ┌─────────────────────────────────────┐
                  │         q_proj Linear Layer         │
                  │                                     │
 x ───────────────┼─── x @ W_q ──────────────────┐      │
 [B, L, 2048]     │   (frozen weights)           │      │
                  │                              │      │
                  │      ┌──── HOOK ────┐        │      │
                  │      │              │        ▼      │
                  │      │  x @ A @ B   │──────(+)──────┼───▶ Q output
                  │      │  (LoRA Δ)    │       │       │     [B, L, out]
                  │      └──────────────┘       │       │
                  │                              │       │
                  └──────────────────────────────┘       │
                                                          
 A = [2048, 16]  ← generated by Hypernetwork
 B = [16, 4096]  ← generated by Hypernetwork
```

### The Bypass Mental Model

Think of LoRA as adding a **bypass circuit** to each linear projection:

```
                    ┌────────────────────┐
                    │   Original Path    │
    Input ─────────►│   W (frozen)       │──────┐
                    └────────────────────┘      │
                                                 ▼
                    ┌────────────────────┐      (+) ────► Output
                    │   LoRA Bypass      │       │
    Input ─────────►│   A @ B (learned)  │──────┘
                    └────────────────────┘
```

- The **original path** (frozen weights) always runs
- The **LoRA bypass** adds a small correction
- The Hypernetwork **generates** what that correction should be, based on the task
- This happens **at runtime** for each sample (task-adaptive LoRA)

---

## Part 4: Hypernetwork → LoRA → Attention Flow

The **Hypernetwork** is a small neural network that **generates** the `A` and `B` matrices based on the prompt:

```
1. Prompt text → tokenize → embed → PromptEncoder → "context vector"
         │
         ▼
2. ShapeGroupedLoRAGenerator:
   - Takes context + per-layer embeddings
   - Outputs 192 pairs of (A, B) matrices (one per target layer)
         │
         ▼
3. lora_dict = {
     "model.layers.0.self_attn.q_proj": (A₀, B₀),
     "model.layers.0.self_attn.k_proj": (A₁, B₁),
     ...
   }
         │
         ▼
4. HookBasedLoRAInjector registers hooks on each target layer
         │
         ▼
5. Student model forward pass → hooks fire → LoRA deltas added
```

### Target Layers in Qwen3-Coder-Next

The system scans for all modules named `q_proj`, `k_proj`, `v_proj`, `o_proj`.

For Qwen3-Coder-Next 80B (48 layers, 4 projections each = **192 targets**):

| Shape | Layers | Description |
|-------|--------|-------------|
| (2048, 2048) | 72 | DeltaNet q, k |
| (2048, 4096) | 48 | DeltaNet v, Attention q |
| (4096, 2048) | 48 | DeltaNet o, Attention o |
| (2048, 512) | 24 | Attention k, v (GQA) |

### How LoRA Modifies Attention

By modifying these projections, LoRA changes:

```
x₁ @ W_q → becomes → x₁ @ W_q + (x₁ @ A_q @ B_q)   ← LoRA modifies Q projection
x₁ @ W_k → becomes → x₁ @ W_k + (x₁ @ A_k @ B_k)   ← LoRA modifies K projection
x₁ @ W_v → becomes → x₁ @ W_v + (x₁ @ A_v @ B_v)   ← LoRA modifies V projection
o₁ @ W_o → becomes → o₁ @ W_o + (o₁ @ A_o @ B_o)   ← LoRA modifies O projection
```

- **Q/K modifications**: Change what tokens attend to (changes similarity scores)
- **V modifications**: Change what information gets retrieved
- **O modifications**: Change how attended output is transformed

This lets the Hypernetwork **steer the model's attention patterns** based on the task!

---

## Part 5: Why Forward Hooks?

The hook-based approach provides:

1. **Non-invasive**: Doesn't modify the model's forward function
2. **Autograd-safe**: Gradients flow through the `+` naturally
3. **FlashAttention compatible**: Works with fused kernels
4. **Temporary**: Hooks are removed after each forward pass

The batched version (`make_batched_lora_hook`) uses `einsum` to apply **different LoRA weights to each sample in a batch**, enabling parallel processing:

```python
def make_batched_lora_hook(lora_A, lora_B, scaling):
    def hook(module, input, output):
        x = input[0]  # [B, L, in_features]
        
        # Batched LoRA: each sample gets its own (A, B) matrices
        # intermediate[b,l,r] = sum_d x[b,l,d] * A[b,d,r]
        intermediate = torch.einsum('bld,bdr->blr', x, lora_A)
        # delta[b,l,o] = sum_r intermediate[b,l,r] * B[b,r,o]
        lora_delta = torch.einsum('blr,bro->blo', intermediate, lora_B) * scaling
        
        return output + lora_delta
    return hook
```

---

## Summary

1. **Attention** lets each token gather information from all other tokens via Q/K similarity → weighted sum of V
2. **LoRA** adds a low-rank bypass `A @ B` to each projection without modifying frozen weights
3. **Forward hooks** intercept layer outputs and add the LoRA delta cleanly
4. **The Hypernetwork** generates task-specific LoRA weights based on the prompt
5. This enables **dynamic, prompt-adaptive attention steering** at runtime
