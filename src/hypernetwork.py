"""
Hypernetwork Architecture for Dynamic LoRA Generation - v2.

SIMPLIFIED based on research findings:
- Both DeltaNet and Attention layers use the SAME projection names (q_proj, k_proj, v_proj, o_proj)
- The difference is layer INDEX not layer NAME
- Single unified generator instead of separate attention/deltanet generators

This version:
1. Uses a single LoRAGenerator for ALL 48 layers
2. Generates LoRAs for all 4 projections per layer
3. Produces a dict mapping layer paths to (A, B) matrices
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        # All layers use the same projection names
        target_modules: Optional[List[str]] = None,
        # Model architecture
        num_layers: int = 48,
        hidden_dim: int = 2048,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank
        
        # Unified targets - same for all layer types
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Total number of LoRA matrices to generate
        self.total_lora_count = num_layers * len(self.target_modules)


class PromptEncoder(nn.Module):
    """
    Encodes the prompt tokens into a fixed-size representation.
    
    Uses attention pooling to aggregate variable-length prompt into
    a fixed context vector that captures task intent.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Transformer encoder for context understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling to get fixed-size output
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: [B, L, D] - Prompt token embeddings
            prompt_mask: [B, L] - Attention mask (1 = attend, 0 = ignore)
            
        Returns:
            context: [B, D] - Pooled context vector
        """
        B = prompt_embeds.shape[0]
        
        # Create key_padding_mask (True = ignore, opposite of attention_mask)
        key_padding_mask = None
        if prompt_mask is not None:
            key_padding_mask = prompt_mask == 0
        
        # Encode prompt sequence
        encoded = self.encoder(prompt_embeds, src_key_padding_mask=key_padding_mask)
        
        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(
            query, encoded, encoded,
            key_padding_mask=key_padding_mask,
        )
        
        context = self.layer_norm(pooled.squeeze(1))
        return context


class ShapeGroupedLoRAGenerator(nn.Module):
    """
    LoRA generator with per-shape output heads.
    
    For Qwen3-Coder-Next, handles 4 unique (in_features, out_features) shapes:
    - (2048, 2048): DeltaNet q_proj, k_proj
    - (2048, 4096): DeltaNet v_proj, Attention q_proj
    - (4096, 2048): DeltaNet o_proj, Attention o_proj
    - (2048, 512): Attention k_proj, v_proj
    
    Architecture:
    1. Shared backbone: context + layer_embedding -> features
    2. Per-shape output heads: features -> (lora_A, lora_B) with correct dims
    """
    
    def __init__(
        self,
        context_dim: int = 2048,
        lora_rank: int = 16,
        layer_shapes: List[Tuple[str, int, int]] = None,  # [(name, in_feat, out_feat), ...]
        zero_init: bool = True,
    ):
        """
        Args:
            context_dim: Dimension of context vector from prompt encoder
            lora_rank: Rank of LoRA matrices
            layer_shapes: List of (layer_name, in_features, out_features) for each target
            zero_init: Initialize B heads to zero for stable training start
        """
        super().__init__()
        self.context_dim = context_dim
        self.lora_rank = lora_rank
        
        if layer_shapes is None:
            raise ValueError("layer_shapes must be provided (from discover_target_layers)")
        
        # Store layer info
        self.num_layers = len(layer_shapes)
        self.layer_names = [ls[0] if isinstance(ls, (tuple, list)) else ls.name for ls in layer_shapes]
        self.layer_shapes = [(ls[1], ls[2]) if isinstance(ls, (tuple, list)) else (ls.in_features, ls.out_features) for ls in layer_shapes]
        
        # Identify unique shapes and create mapping
        unique_shapes = list(set(self.layer_shapes))
        self.unique_shapes = unique_shapes
        self.shape_to_idx = {shape: i for i, shape in enumerate(unique_shapes)}
        self.layer_to_shape_idx = [self.shape_to_idx[s] for s in self.layer_shapes]
        
        # Learnable layer embeddings (one per target layer)
        self.layer_embed = nn.Embedding(self.num_layers, context_dim)
        
        # Shared generator backbone
        self.generator = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        
        # Per-shape output heads
        self.heads_A = nn.ModuleDict()
        self.heads_B = nn.ModuleDict()
        
        for in_dim, out_dim in unique_shapes:
            shape_key = f"{in_dim}_{out_dim}"
            # A: projects input (in_dim) down to rank
            self.heads_A[shape_key] = nn.Linear(context_dim, in_dim * lora_rank)
            # B: projects rank up to output (out_dim)
            self.heads_B[shape_key] = nn.Linear(context_dim, lora_rank * out_dim)
            
            if zero_init:
                nn.init.zeros_(self.heads_B[shape_key].weight)
                nn.init.zeros_(self.heads_B[shape_key].bias)
                nn.init.normal_(self.heads_A[shape_key].weight, std=0.01)
                nn.init.zeros_(self.heads_A[shape_key].bias)
        
        # Pre-compute shape keys for efficiency
        self._shape_keys = [f"{s[0]}_{s[1]}" for s in self.layer_shapes]
    
    def forward(
        self,
        context: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate LoRA matrices for all layers.
        
        Args:
            context: [B, D] - Context from prompt encoder
            
        Returns:
            Dict mapping layer_name -> (lora_A, lora_B)
            where lora_A: [B, in_features, rank]
                  lora_B: [B, rank, out_features]
        """
        B = context.shape[0]
        device = context.device
        
        # Get all layer embeddings
        layer_indices = torch.arange(self.num_layers, device=device)
        layer_emb = self.layer_embed(layer_indices)  # [N, D]
        
        # Expand context for all layers
        context_expanded = context.unsqueeze(1).expand(-1, self.num_layers, -1)  # [B, N, D]
        layer_emb_expanded = layer_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # Generate features through shared backbone
        generator_input = torch.cat([context_expanded, layer_emb_expanded], dim=-1)  # [B, N, 2D]
        features = self.generator(generator_input)  # [B, N, D]
        
        # Apply per-shape heads
        result = {}
        for i, layer_name in enumerate(self.layer_names):
            shape_key = self._shape_keys[i]
            in_dim, out_dim = self.layer_shapes[i]
            
            feat = features[:, i, :]  # [B, D]
            
            lora_A_flat = self.heads_A[shape_key](feat)  # [B, in_dim * rank]
            lora_B_flat = self.heads_B[shape_key](feat)  # [B, rank * out_dim]
            
            lora_A = lora_A_flat.view(B, in_dim, self.lora_rank)  # [B, in_dim, rank]
            lora_B = lora_B_flat.view(B, self.lora_rank, out_dim)  # [B, rank, out_dim]
            
            result[layer_name] = (lora_A, lora_B)
        
        return result


# Keep old class for backwards compatibility
class UnifiedLoRAGenerator(ShapeGroupedLoRAGenerator):
    """Legacy alias - use ShapeGroupedLoRAGenerator instead."""
    
    def __init__(
        self,
        context_dim: int = 2048,
        hidden_dim: int = 2048,
        lora_rank: int = 16,
        num_layers: int = 48,
        num_modules: int = 4,
        zero_init: bool = True,
    ):
        # Create mock layer shapes with uniform dimensions (legacy behavior)
        import warnings
        warnings.warn(
            "UnifiedLoRAGenerator is deprecated - it uses uniform dimensions which "
            "is incorrect for Qwen3-Coder-Next. Use ShapeGroupedLoRAGenerator instead.",
            DeprecationWarning,
        )
        
        layer_shapes = [
            (f"layer_{i}_module_{j}", hidden_dim, hidden_dim)
            for i in range(num_layers)
            for j in range(num_modules)
        ]
        super().__init__(
            context_dim=context_dim,
            lora_rank=lora_rank,
            layer_shapes=layer_shapes,
            zero_init=zero_init,
        )


class AgenticHyperNetwork(nn.Module):
    """
    Main Hypernetwork that generates dynamic LoRA adapters.
    
    v4 Fixes:
    - Uses ShapeGroupedLoRAGenerator with per-shape output heads
    - Handles varying (in_features, out_features) for DeltaNet/Attention layers
    - Supports Qwen3-Coder-Next's hybrid Gated DeltaNet + Gated Attention architecture
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        lora_config: Optional[LoRAConfig] = None,
        dropout: float = 0.1,
        # CRITICAL: Pass LayerInfo list with dimensions
        target_layer_names: Optional[List] = None,  # List[LayerInfo] or List[str]
    ):
        super().__init__()
        
        self.lora_config = lora_config or LoRAConfig()
        self.hidden_dim = hidden_dim
        
        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Process target_layer_names to extract shapes
        if target_layer_names is None:
            raise ValueError(
                "target_layer_names must be provided (from discover_target_layers). "
                "This ensures LoRA matrices have correct dimensions for each layer."
            )
        
        # Handle both LayerInfo and plain strings (for backwards compat)
        if len(target_layer_names) > 0:
            first = target_layer_names[0]
            if hasattr(first, 'in_features'):
                # It's LayerInfo
                self.layer_shapes = target_layer_names
                self.target_layer_names = [li.name for li in target_layer_names]
            elif isinstance(first, (tuple, list)) and len(first) == 3:
                # It's (name, in_features, out_features) tuple
                self.layer_shapes = target_layer_names
                self.target_layer_names = [t[0] for t in target_layer_names]
            else:
                # Plain strings - fallback to uniform dimensions (legacy)
                import warnings
                warnings.warn(
                    "target_layer_names contains plain strings without dimensions. "
                    "This may cause dimension mismatches. Use discover_target_layers() "
                    "with return_dimensions=True for correct shapes.",
                    DeprecationWarning,
                )
                self.target_layer_names = target_layer_names
                self.layer_shapes = [
                    (name, hidden_dim, hidden_dim) for name in target_layer_names
                ]
        else:
            raise ValueError("target_layer_names cannot be empty")
        
        self.num_loras = len(self.layer_shapes)
        
        # Shape-grouped generator with per-shape output heads
        self.lora_generator = ShapeGroupedLoRAGenerator(
            context_dim=hidden_dim,
            lora_rank=self.lora_config.rank,
            layer_shapes=self.layer_shapes,
            zero_init=True,
        )
        
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA weights for all target layers.
        
        Args:
            prompt_embeds: [B, L, D] - Prompt token embeddings
            prompt_mask: [B, L] - Attention mask
            
        Returns:
            Dict with:
                "lora_dict": Dict[layer_name -> (lora_A, lora_B)]
                "context": [B, D] - For debugging
        """
        context = self.prompt_encoder(prompt_embeds, prompt_mask)
        lora_dict = self.lora_generator(context)  # Dict[name -> (A, B)]
        
        return {
            "lora_dict": lora_dict,
            "context": context,
        }
    
    def get_lora_dict(
        self,
        lora_output: Dict[str, any],
        batch_idx: int = 0,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract LoRA dict for a single batch element.
        
        Args:
            lora_output: Output from forward()
            batch_idx: Which batch element to extract
            
        Returns:
            Dict mapping layer paths to (A, B) tuples
            e.g., {"model.layers.0.self_attn.q_proj": (A, B), 
                   "model.layers.1.deltanet.q_proj": (A, B), ...}
        """
        lora_dict = lora_output["lora_dict"]
        
        result = {}
        for name, (lora_A, lora_B) in lora_dict.items():
            # Extract single batch element
            result[name] = (lora_A[batch_idx], lora_B[batch_idx])
        
        return result
    
    def get_lora_dict_batched(
        self,
        lora_output: Dict[str, any],
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get LoRA dicts for all samples in batch."""
        # Get batch size from first layer's A matrix
        lora_dict = lora_output["lora_dict"]
        first_name = next(iter(lora_dict.keys()))
        batch_size = lora_dict[first_name][0].shape[0]
        return [self.get_lora_dict(lora_output, b) for b in range(batch_size)]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        return {
            "prompt_encoder": sum(p.numel() for p in self.prompt_encoder.parameters()),
            "lora_generator": sum(p.numel() for p in self.lora_generator.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
    
    def estimate_lora_params(self) -> int:
        """Estimate total LoRA parameters generated per forward pass."""
        rank = self.lora_config.rank
        total = 0
        for (in_dim, out_dim) in set(self.lora_generator.layer_shapes):
            count = sum(1 for s in self.lora_generator.layer_shapes if s == (in_dim, out_dim))
            per_lora = in_dim * rank + rank * out_dim  # A + B
            total += per_lora * count
        return total


def test_hypernetwork():
    """Test the shape-grouped hypernetwork with Qwen3-Coder-Next shapes."""
    print("=" * 60)
    print("Testing Shape-Grouped Hypernetwork v4")
    print("=" * 60)
    
    config = LoRAConfig(rank=16, alpha=32, num_layers=48, hidden_dim=2048)
    
    # Generate realistic layer info for Qwen3-Coder-Next
    # 36 DeltaNet layers (indices 0,1,2, 4,5,6, ...) + 12 Attention layers (indices 3, 7, 11, ...)
    layer_shapes = []
    for i in range(48):
        is_attention = (i % 4 == 3)  # Every 4th layer is Attention
        
        if is_attention:
            # GatedAttention: q=4096, k/v=512, o=4096->2048
            layer_shapes.append((f"model.layers.{i}.self_attn.q_proj", 2048, 4096))
            layer_shapes.append((f"model.layers.{i}.self_attn.k_proj", 2048, 512))
            layer_shapes.append((f"model.layers.{i}.self_attn.v_proj", 2048, 512))
            layer_shapes.append((f"model.layers.{i}.self_attn.o_proj", 4096, 2048))
        else:
            # GatedDeltaNet: q/k=2048, v=4096, o=4096->2048
            layer_shapes.append((f"model.layers.{i}.deltanet.q_proj", 2048, 2048))
            layer_shapes.append((f"model.layers.{i}.deltanet.k_proj", 2048, 2048))
            layer_shapes.append((f"model.layers.{i}.deltanet.v_proj", 2048, 4096))
            layer_shapes.append((f"model.layers.{i}.deltanet.o_proj", 4096, 2048))
    
    print(f"\nTotal target layers: {len(layer_shapes)}")
    
    # Count unique shapes
    unique_shapes = set((s[1], s[2]) for s in layer_shapes)
    print(f"Unique shapes: {len(unique_shapes)}")
    for shape in sorted(unique_shapes):
        count = sum(1 for s in layer_shapes if (s[1], s[2]) == shape)
        print(f"  {shape}: {count} layers")
    
    hypernet = AgenticHyperNetwork(
        hidden_dim=2048,
        num_encoder_layers=4,
        num_heads=8,
        lora_config=config,
        target_layer_names=layer_shapes,
    )
    
    # Parameter counts
    param_counts = hypernet.count_parameters()
    print(f"\nParameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,} ({v * 4 / 1e6:.1f} MB)")
    
    print(f"\nLoRA params per forward: {hypernet.estimate_lora_params():,}")
    
    # Test forward
    batch_size = 2
    seq_len = 512
    
    prompt_embeds = torch.randn(batch_size, seq_len, 2048)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    output = hypernet(prompt_embeds, prompt_mask)
    
    print(f"\nOutput:")
    print(f"  lora_dict: {len(output['lora_dict'])} entries")
    print(f"  context: {output['context'].shape}")
    
    # Check a sample of shapes
    print(f"\nSample LoRA shapes:")
    sample_keys = list(output['lora_dict'].keys())[:4]
    for key in sample_keys:
        A, B = output['lora_dict'][key]
        print(f"  {key.split('.')[-1]}: A={list(A.shape)}, B={list(B.shape)}")
    
    # Test dict extraction
    lora_dict = hypernet.get_lora_dict(output, batch_idx=0)
    print(f"\nExtracted LoRA dict has {len(lora_dict)} entries")
    
    # Test gradient flow
    loss = sum(A.sum() + B.sum() for A, B in output['lora_dict'].values())
    loss.backward()
    
    has_grad = hypernet.prompt_encoder.pool_query.grad is not None
    print(f"\nGradient flow: {'✓ PASSED' if has_grad else '✗ FAILED'}")
    
    # Test zero-init (all B matrices should be near zero)
    b_means = [B.abs().mean().item() for _, B in output['lora_dict'].values()]
    b_mean = sum(b_means) / len(b_means)
    print(f"B matrix mean abs: {b_mean:.6f} (should be ~0)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_hypernetwork()
