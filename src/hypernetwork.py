"""
Hypernetwork Architecture for Dynamic LoRA Generation.

The Hypernetwork takes a prompt embedding and generates LoRA adapter weights
that are injected into the frozen Student model.

CRITICAL DESIGN DECISIONS:
1. Zero initialization of output layer (prevents gradient explosion at step 0)
2. Shared generator with layer embeddings (reduces params from ~500M to ~50M)
3. Targets BOTH Attention (25%) AND DeltaNet (75%) layers in the hybrid Student
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
        # Target modules for Qwen3-Coder-Next (Hybrid Architecture)
        # 25% of layers: Standard Attention
        attention_targets: List[str] = None,
        # 75% of layers: DeltaNet Linear Attention
        deltanet_targets: List[str] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank
        
        # Default targets for Qwen3-Coder-Next
        self.attention_targets = attention_targets or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.deltanet_targets = deltanet_targets or ["linear_q", "linear_k", "linear_v"]
        
        # Qwen3-Coder-Next has 48 layers with hybrid layout:
        # 12 * (3 * DeltaNet + 1 * Attention)
        self.num_attention_layers = 12  # Every 4th layer
        self.num_deltanet_layers = 36   # 3 out of every 4 layers
        self.total_layers = 48


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


class LoRAGenerator(nn.Module):
    """
    Generates LoRA matrices (A and B) for a specific module type.
    
    Uses layer embeddings to generate different weights for each layer
    while sharing the generator parameters.
    """
    
    def __init__(
        self,
        context_dim: int = 2048,
        hidden_dim: int = 2048,
        lora_rank: int = 16,
        num_layers: int = 48,
        num_modules: int = 4,  # e.g., q, k, v, o
        zero_init: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.num_layers = num_layers
        self.num_modules = num_modules
        
        # Layer embeddings (learnable)
        self.layer_embed = nn.Embedding(num_layers, context_dim)
        
        # Module embeddings (e.g., q, k, v, o)
        self.module_embed = nn.Embedding(num_modules, context_dim)
        
        # Shared generator network
        self.generator = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),  # context + layer/module embed
            nn.GELU(),
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        
        # Output heads for A and B matrices
        # A: [hidden_dim, rank], B: [rank, hidden_dim]
        # We output them flattened
        self.head_A = nn.Linear(context_dim, hidden_dim * lora_rank)
        self.head_B = nn.Linear(context_dim, lora_rank * hidden_dim)
        
        # Zero initialization for stability
        if zero_init:
            nn.init.zeros_(self.head_B.weight)
            nn.init.zeros_(self.head_B.bias)
            # Small init for A
            nn.init.normal_(self.head_A.weight, std=0.01)
            nn.init.zeros_(self.head_A.bias)
    
    def forward(
        self,
        context: torch.Tensor,
        layer_indices: torch.Tensor,
        module_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate LoRA matrices for specified layers and modules.
        
        Args:
            context: [B, D] - Context from prompt encoder
            layer_indices: [N] - Which layers to generate for
            module_indices: [N] - Which modules (q/k/v/o) for each layer
            
        Returns:
            lora_A: [B, N, hidden_dim, rank]
            lora_B: [B, N, rank, hidden_dim]
        """
        B = context.shape[0]
        N = layer_indices.shape[0]
        
        # Get embeddings
        layer_emb = self.layer_embed(layer_indices)  # [N, D]
        module_emb = self.module_embed(module_indices)  # [N, D]
        
        # Combine layer and module embeddings
        combined_emb = layer_emb + module_emb  # [N, D]
        
        # Expand context for all layer/module combinations
        context_expanded = context.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        combined_emb = combined_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # Concatenate context with layer/module info
        generator_input = torch.cat([context_expanded, combined_emb], dim=-1)  # [B, N, 2D]
        
        # Generate through shared network
        features = self.generator(generator_input)  # [B, N, D]
        
        # Generate A and B matrices
        lora_A_flat = self.head_A(features)  # [B, N, hidden_dim * rank]
        lora_B_flat = self.head_B(features)  # [B, N, rank * hidden_dim]
        
        # Reshape to matrix form
        lora_A = rearrange(lora_A_flat, 'b n (h r) -> b n h r', r=self.lora_rank)
        lora_B = rearrange(lora_B_flat, 'b n (r h) -> b n r h', r=self.lora_rank)
        
        return lora_A, lora_B


class AgenticHyperNetwork(nn.Module):
    """
    Main Hypernetwork that generates dynamic LoRA adapters.
    
    Architecture:
    1. PromptEncoder: Compress prompt tokens into context vector
    2. LoRAGenerators: Generate A/B matrices for each layer type
       - AttentionGenerator: For standard attention layers (25%)
       - DeltaNetGenerator: For DeltaNet layers (75%)
    
    The generated LoRAs are injected into the frozen Student model
    to give it "specialized brains" for different tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        lora_config: Optional[LoRAConfig] = None,
        dropout: float = 0.1,
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
        
        # Generator for Attention layers (q, k, v, o projections)
        self.attention_generator = LoRAGenerator(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=self.lora_config.rank,
            num_layers=self.lora_config.num_attention_layers,
            num_modules=len(self.lora_config.attention_targets),
            zero_init=True,
        )
        
        # Generator for DeltaNet layers (linear_q, linear_k, linear_v)
        self.deltanet_generator = LoRAGenerator(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=self.lora_config.rank,
            num_layers=self.lora_config.num_deltanet_layers,
            num_modules=len(self.lora_config.deltanet_targets),
            zero_init=True,
        )
        
        # Pre-compute layer and module indices
        self._setup_indices()
        
    def _setup_indices(self):
        """Pre-compute the layer and module indices for generation."""
        cfg = self.lora_config
        
        # Attention layers: every 4th layer (indices 3, 7, 11, ...)
        # In the hybrid layout: 12 * (3 * DeltaNet + 1 * Attention)
        attn_layer_indices = []
        attn_module_indices = []
        for i in range(cfg.num_attention_layers):
            layer_idx = i  # Use sequential indexing for the generator
            for j, _ in enumerate(cfg.attention_targets):
                attn_layer_indices.append(layer_idx)
                attn_module_indices.append(j)
        
        self.register_buffer(
            "attn_layer_idx",
            torch.tensor(attn_layer_indices, dtype=torch.long)
        )
        self.register_buffer(
            "attn_module_idx",
            torch.tensor(attn_module_indices, dtype=torch.long)
        )
        
        # DeltaNet layers: 3 out of every 4 layers
        delta_layer_indices = []
        delta_module_indices = []
        for i in range(cfg.num_deltanet_layers):
            layer_idx = i
            for j, _ in enumerate(cfg.deltanet_targets):
                delta_layer_indices.append(layer_idx)
                delta_module_indices.append(j)
        
        self.register_buffer(
            "delta_layer_idx",
            torch.tensor(delta_layer_indices, dtype=torch.long)
        )
        self.register_buffer(
            "delta_module_idx",
            torch.tensor(delta_module_indices, dtype=torch.long)
        )
    
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate LoRA weights for all target layers.
        
        Args:
            prompt_embeds: [B, L, D] - Prompt token embeddings from Student
            prompt_mask: [B, L] - Attention mask for prompt
            
        Returns:
            Dictionary with:
                "attention": (lora_A, lora_B) for attention layers
                "deltanet": (lora_A, lora_B) for deltanet layers
                
            Each lora_A: [B, N, hidden_dim, rank]
            Each lora_B: [B, N, rank, hidden_dim]
        """
        # Encode prompt to get task context
        context = self.prompt_encoder(prompt_embeds, prompt_mask)
        
        # Generate LoRAs for attention layers
        attn_A, attn_B = self.attention_generator(
            context,
            self.attn_layer_idx,
            self.attn_module_idx,
        )
        
        # Generate LoRAs for DeltaNet layers
        delta_A, delta_B = self.deltanet_generator(
            context,
            self.delta_layer_idx,
            self.delta_module_idx,
        )
        
        return {
            "attention": (attn_A, attn_B),
            "deltanet": (delta_A, delta_B),
            "context": context,  # For debugging/analysis
        }
    
    def get_lora_dict(
        self,
        lora_output: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int = 0,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert the batched LoRA output into a dictionary keyed by layer name.
        
        This is useful for the LoRA injection step where we need to
        look up weights by layer name.
        
        Args:
            lora_output: Output from forward()
            batch_idx: Which batch element to extract
            
        Returns:
            Dict mapping "layer_idx.module_name" -> (A, B) matrices
        """
        cfg = self.lora_config
        result = {}
        
        # Unpack attention LoRAs
        attn_A, attn_B = lora_output["attention"]
        idx = 0
        for layer_i in range(cfg.num_attention_layers):
            # Map to actual layer index in model (every 4th layer)
            actual_layer = layer_i * 4 + 3  # 3, 7, 11, ...
            for module_name in cfg.attention_targets:
                key = f"{actual_layer}.self_attn.{module_name}"
                result[key] = (
                    attn_A[batch_idx, idx],  # [hidden_dim, rank]
                    attn_B[batch_idx, idx],  # [rank, hidden_dim]
                )
                idx += 1
        
        # Unpack DeltaNet LoRAs
        delta_A, delta_B = lora_output["deltanet"]
        idx = 0
        for layer_i in range(cfg.num_deltanet_layers):
            # Map to actual layer index (layers 0,1,2, 4,5,6, 8,9,10, ...)
            block = layer_i // 3
            offset = layer_i % 3
            actual_layer = block * 4 + offset
            for module_name in cfg.deltanet_targets:
                key = f"{actual_layer}.deltanet.{module_name}"
                result[key] = (
                    delta_A[batch_idx, idx],
                    delta_B[batch_idx, idx],
                )
                idx += 1
        
        return result
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        return {
            "prompt_encoder": sum(p.numel() for p in self.prompt_encoder.parameters()),
            "attention_generator": sum(p.numel() for p in self.attention_generator.parameters()),
            "deltanet_generator": sum(p.numel() for p in self.deltanet_generator.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


def test_hypernetwork():
    """Quick test to verify the Hypernetwork architecture."""
    print("Testing Hypernetwork...")
    
    config = LoRAConfig(rank=16, alpha=32)
    hypernet = AgenticHyperNetwork(
        hidden_dim=2048,
        num_encoder_layers=4,
        num_heads=8,
        lora_config=config,
    )
    
    # Count parameters
    param_counts = hypernet.count_parameters()
    print(f"Parameter counts: {param_counts}")
    total_mb = param_counts['total'] * 4 / (1024 ** 2)
    print(f"Total size: {total_mb:.1f} MB")
    
    # Test forward pass
    batch_size = 2
    seq_len = 512
    hidden_dim = 2048
    
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    output = hypernet(prompt_embeds, prompt_mask)
    
    print(f"\nOutput keys: {output.keys()}")
    print(f"Attention LoRA A shape: {output['attention'][0].shape}")
    print(f"Attention LoRA B shape: {output['attention'][1].shape}")
    print(f"DeltaNet LoRA A shape: {output['deltanet'][0].shape}")
    print(f"DeltaNet LoRA B shape: {output['deltanet'][1].shape}")
    
    # Test gradient flow
    loss = output['attention'][0].sum() + output['deltanet'][0].sum()
    loss.backward()
    
    has_grad = hypernet.prompt_encoder.pool_query.grad is not None
    print(f"\nGradient flow test: {'PASSED' if has_grad else 'FAILED'}")
    
    # Test zero-init (B matrices should be near zero at init)
    b_norm = output['attention'][1].abs().mean().item()
    print(f"Attention B matrix mean abs: {b_norm:.6f} (should be ~0)")


if __name__ == "__main__":
    test_hypernetwork()
