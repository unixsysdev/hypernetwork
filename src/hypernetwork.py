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


class UnifiedLoRAGenerator(nn.Module):
    """
    Single generator for ALL layers.
    
    Uses layer + module embeddings to produce different LoRAs
    for each (layer, module) combination.
    """
    
    def __init__(
        self,
        context_dim: int = 2048,
        hidden_dim: int = 2048,
        lora_rank: int = 16,
        num_layers: int = 48,
        num_modules: int = 4,  # q, k, v, o
        zero_init: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.num_layers = num_layers
        self.num_modules = num_modules
        
        # Learnable embeddings
        self.layer_embed = nn.Embedding(num_layers, context_dim)
        self.module_embed = nn.Embedding(num_modules, context_dim)
        
        # Shared generator network
        self.generator = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        
        # Output heads for A and B matrices
        self.head_A = nn.Linear(context_dim, hidden_dim * lora_rank)
        self.head_B = nn.Linear(context_dim, lora_rank * hidden_dim)
        
        # Zero initialization for stability
        if zero_init:
            nn.init.zeros_(self.head_B.weight)
            nn.init.zeros_(self.head_B.bias)
            nn.init.normal_(self.head_A.weight, std=0.01)
            nn.init.zeros_(self.head_A.bias)
    
    def forward(
        self,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA matrices for ALL layers and modules.
        
        Args:
            context: [B, D] - Context from prompt encoder
            
        Returns:
            lora_A: [B, num_layers * num_modules, hidden_dim, rank]
            lora_B: [B, num_layers * num_modules, rank, hidden_dim]
        """
        B = context.shape[0]
        device = context.device
        
        # Create indices for all (layer, module) combinations
        # Total: num_layers * num_modules
        layer_indices = torch.arange(self.num_layers, device=device).repeat_interleave(self.num_modules)
        module_indices = torch.arange(self.num_modules, device=device).repeat(self.num_layers)
        
        N = layer_indices.shape[0]  # num_layers * num_modules
        
        # Get embeddings
        layer_emb = self.layer_embed(layer_indices)  # [N, D]
        module_emb = self.module_embed(module_indices)  # [N, D]
        combined_emb = layer_emb + module_emb  # [N, D]
        
        # Expand for batch
        context_expanded = context.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        combined_emb = combined_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # Generate
        generator_input = torch.cat([context_expanded, combined_emb], dim=-1)
        features = self.generator(generator_input)
        
        lora_A_flat = self.head_A(features)
        lora_B_flat = self.head_B(features)
        
        lora_A = rearrange(lora_A_flat, 'b n (h r) -> b n h r', r=self.lora_rank)
        lora_B = rearrange(lora_B_flat, 'b n (r h) -> b n r h', r=self.lora_rank)
        
        return lora_A, lora_B


class AgenticHyperNetwork(nn.Module):
    """
    Main Hypernetwork that generates dynamic LoRA adapters.
    
    v3 Fixes:
    - Uses DISCOVERED layer names, not template-based guessing
    - Guarantees correct mapping for both Attention AND DeltaNet layers
    - Single unified generator for all layers
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        lora_config: Optional[LoRAConfig] = None,
        dropout: float = 0.1,
        # CRITICAL: Pass actual discovered layer names
        target_layer_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.lora_config = lora_config or LoRAConfig()
        self.hidden_dim = hidden_dim
        self.target_layer_names = target_layer_names  # e.g., ["model.layers.0.self_attn.q_proj", ...]
        
        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Determine number of LoRAs to generate
        if target_layer_names is not None:
            num_loras = len(target_layer_names)
        else:
            num_loras = self.lora_config.num_layers * len(self.lora_config.target_modules)
        
        self.num_loras = num_loras
        
        # Single unified generator - sized to match discovered layers
        self.lora_generator = UnifiedLoRAGenerator(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=self.lora_config.rank,
            num_layers=num_loras,  # One embedding per discovered layer
            num_modules=1,  # Flatten - each layer+module is one entry
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
                "lora_A": [B, num_layers * num_modules, hidden_dim, rank]
                "lora_B": [B, num_layers * num_modules, rank, hidden_dim]
                "context": [B, D] - For debugging
        """
        context = self.prompt_encoder(prompt_embeds, prompt_mask)
        lora_A, lora_B = self.lora_generator(context)
        
        return {
            "lora_A": lora_A,
            "lora_B": lora_B,
            "context": context,
        }
    
    def get_lora_dict(
        self,
        lora_output: Dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert batched output to dict keyed by ACTUAL layer paths.
        
        CRITICAL: Uses discovered layer names (ordered mapping).
        The i-th generated LoRA goes to the i-th discovered layer.
        This guarantees correct mapping for BOTH Attention AND DeltaNet layers.
        
        Args:
            lora_output: Output from forward()
            batch_idx: Which batch element to extract
            
        Returns:
            Dict mapping layer paths to (A, B) tuples
            e.g., {"model.layers.0.self_attn.q_proj": (A, B), 
                   "model.layers.1.deltanet.q_proj": (A, B), ...}
        """
        lora_A = lora_output["lora_A"][batch_idx]  # [N, hidden_dim, rank]
        lora_B = lora_output["lora_B"][batch_idx]  # [N, rank, hidden_dim]
        
        # CRITICAL: Use actual discovered names, NOT template
        if self.target_layer_names is None:
            raise ValueError(
                "target_layer_names must be provided to AgenticHyperNetwork. "
                "Use discover_target_layers() to get the real layer names."
            )
        
        if len(self.target_layer_names) != lora_A.shape[0]:
            raise ValueError(
                f"Mismatch: Generated {lora_A.shape[0]} LoRAs but have "
                f"{len(self.target_layer_names)} target layers. "
                f"This likely means the model architecture changed."
            )
        
        result = {}
        for idx, name in enumerate(self.target_layer_names):
            result[name] = (lora_A[idx], lora_B[idx])
        
        return result
    
    def get_lora_dict_batched(
        self,
        lora_output: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Get LoRA dicts for all samples in batch."""
        batch_size = lora_output["lora_A"].shape[0]
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
        cfg = self.lora_config
        per_lora = 2 * cfg.hidden_dim * cfg.rank  # A + B
        return per_lora * cfg.num_layers * len(cfg.target_modules)


def test_hypernetwork():
    """Test the simplified hypernetwork."""
    print("=" * 60)
    print("Testing Simplified Hypernetwork v2")
    print("=" * 60)
    
    config = LoRAConfig(rank=16, alpha=32, num_layers=48, hidden_dim=2048)
    
    # Generate mock layer names to satisfy target_layer_names requirement
    mock_layer_names = [
        f"model.layers.{i}.self_attn.{m}"
        for i in range(48)
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]
    
    hypernet = AgenticHyperNetwork(
        hidden_dim=2048,
        num_encoder_layers=4,
        num_heads=8,
        lora_config=config,
        target_layer_names=mock_layer_names,
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
    
    print(f"\nOutput shapes:")
    print(f"  lora_A: {output['lora_A'].shape}")
    print(f"  lora_B: {output['lora_B'].shape}")
    print(f"  context: {output['context'].shape}")
    
    # Test dict conversion
    lora_dict = hypernet.get_lora_dict(output, batch_idx=0)
    print(f"\nLoRA dict has {len(lora_dict)} entries")
    print(f"Sample keys: {list(lora_dict.keys())[:4]}")
    
    # Test gradient flow
    loss = output['lora_A'].sum() + output['lora_B'].sum()
    loss.backward()
    
    has_grad = hypernet.prompt_encoder.pool_query.grad is not None
    print(f"\nGradient flow: {'✓ PASSED' if has_grad else '✗ FAILED'}")
    
    # Test zero-init
    b_mean = output['lora_B'].abs().mean().item()
    print(f"B matrix mean abs: {b_mean:.6f} (should be ~0)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_hypernetwork()
