"""
LoRA Injection Module v2 - Hook-Based Implementation.

This fixes the fragile forward-method replacement with proper forward hooks.

Key changes:
1. Uses register_forward_hook to ADD LoRA delta (not replace forward)
2. Autograd-safe: gradients flow through the addition naturally
3. Works with fused kernels and any model architecture
"""

from typing import Dict, Tuple, Optional, Callable, List, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get a submodule by its dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and hasattr(module, '__getitem__'):
            module = module[int(part)]
        else:
            return None
    return module


def make_lora_hook(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float = 1.0,
) -> Callable:
    """
    Create a forward hook that ADDS LoRA delta to the output.
    
    This is autograd-safe because:
    - We don't modify the forward function
    - We just add to the output tensor
    - Gradients flow through addition naturally
    
    Args:
        lora_A: [in_features, rank] - Down projection
        lora_B: [rank, out_features] - Up projection  
        scaling: alpha / rank
    
    Returns:
        Hook function compatible with register_forward_hook
    """
    def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        # input is a tuple, first element is the actual input tensor
        x = input[0]  # [batch, seq, in_features]
        
        # LoRA computation: (x @ A) @ B * scaling
        # A: [in_features, rank], B: [rank, out_features]
        lora_delta = (x @ lora_A) @ lora_B * scaling
        
        # ADD to output, don't replace
        return output + lora_delta
    
    return hook


class HookBasedLoRAInjector:
    """
    Manages LoRA injection via forward hooks.
    
    Usage:
        injector = HookBasedLoRAInjector(student_model)
        
        # Apply LoRA for a forward pass
        with injector.apply_lora(lora_dict):
            output = student_model(input_ids)
        # Hooks automatically removed after context
    """
    
    def __init__(
        self,
        model: nn.Module,
        scaling: float = 2.0,  # alpha / rank
    ):
        self.model = model
        self.scaling = scaling
        
    @contextmanager
    def apply_lora(
        self,
        lora_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Context manager to temporarily apply LoRA weights.
        
        Args:
            lora_dict: Maps layer names to (lora_A, lora_B) tuples
                      e.g., {"model.layers.0.self_attn.q_proj": (A, B)}
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []
        
        try:
            # Register hooks for each target layer
            for name, (lora_A, lora_B) in lora_dict.items():
                module = get_module_by_name(self.model, name)
                
                if module is None:
                    # Module not found - log for debugging
                    import logging
                    logging.getLogger(__name__).debug(f"Module not found: {name}")
                    continue
                    
                if not isinstance(module, nn.Linear):
                    # Only apply to Linear layers
                    continue
                
                hook = make_lora_hook(lora_A, lora_B, self.scaling)
                handle = module.register_forward_hook(hook)
                handles.append(handle)
            
            yield  # Run the forward pass
            
        finally:
            # Always remove hooks, even if exception occurs
            for handle in handles:
                handle.remove()


class BatchedLoRAInjector:
    """
    Optimized injector for batched LoRA application.
    
    Instead of different LoRAs per sample, this applies the same LoRA
    to all samples in a batch. Use this when prompt context is similar.
    """
    
    def __init__(
        self,
        model: nn.Module,
        scaling: float = 2.0,
    ):
        self.model = model
        self.scaling = scaling
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._current_lora: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        
    def set_lora(self, lora_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Set the LoRA weights to use for subsequent forward passes."""
        # Remove old hooks
        self.clear_lora()
        
        # Register new hooks
        for name, (lora_A, lora_B) in lora_dict.items():
            module = get_module_by_name(self.model, name)
            if module is not None and isinstance(module, nn.Linear):
                hook = make_lora_hook(lora_A, lora_B, self.scaling)
                handle = module.register_forward_hook(hook)
                self._hook_handles.append(handle)
        
        self._current_lora = lora_dict
        
    def clear_lora(self):
        """Remove all LoRA hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._current_lora = None

# Type for layer discovery results
from dataclasses import dataclass
from typing import NamedTuple

class LayerInfo(NamedTuple):
    """Information about a discovered target layer."""
    name: str
    in_features: int
    out_features: int


def discover_target_layers(
    model: nn.Module,
    target_names: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    return_dimensions: bool = True,
) -> List[LayerInfo]:
    """
    Discover all layers in the model that match target names.
    
    This automatically handles the difference between DeltaNet and Attention
    layers since they both use the same projection names.
    
    For Qwen3-Coder-Next, this discovers 192 layers with 4 unique shapes:
    - (2048, 2048): DeltaNet q_proj, k_proj
    - (2048, 4096): DeltaNet v_proj, Attention q_proj
    - (4096, 2048): DeltaNet o_proj, Attention o_proj
    - (2048, 512): Attention k_proj, v_proj
    
    Args:
        model: The model to discover layers in
        target_names: Names to match (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"])
        return_dimensions: If True, return LayerInfo with dimensions. 
                          If False, just return List[str] for backwards compat.
    
    Returns:
        List of LayerInfo(name, in_features, out_features)
    """
    discovered = []
    
    for name, module in model.named_modules():
        # Check if this module's name ends with a target
        for target in target_names:
            if name.endswith(target) and isinstance(module, nn.Linear):
                if return_dimensions:
                    discovered.append(LayerInfo(
                        name=name,
                        in_features=module.in_features,
                        out_features=module.out_features,
                    ))
                else:
                    discovered.append(name)
                break
    
    return discovered


def group_layers_by_shape(
    layers: List[LayerInfo],
) -> Dict[Tuple[int, int], List[LayerInfo]]:
    """
    Group layers by their (in_features, out_features) shape.
    
    Returns a dict mapping shape tuples to lists of LayerInfo.
    """
    groups = {}
    for layer in layers:
        shape = (layer.in_features, layer.out_features)
        if shape not in groups:
            groups[shape] = []
        groups[shape].append(layer)
    return groups


def create_layer_to_index_mapping(
    layer_paths: List[str],
) -> Dict[str, int]:
    """
    Create a mapping from layer paths to sequential indices.
    
    This is useful for the Hypernetwork which generates LoRAs in a specific order.
    """
    return {path: idx for idx, path in enumerate(sorted(layer_paths))}


# For backwards compatibility
class LoRAInjector(HookBasedLoRAInjector):
    """Alias for backwards compatibility."""
    
    def inject(self, lora_dict):
        """Backwards-compatible interface."""
        return self.apply_lora(lora_dict)


# =============================================================================
# Tests
# =============================================================================

def test_hook_based_injection():
    """Test that hook-based injection works and gradients flow correctly."""
    print("=" * 60)
    print("Testing Hook-Based LoRA Injection")
    print("=" * 60)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            return self.layer2(x)
    
    model = SimpleModel()
    
    # Freeze the model (simulating frozen student)
    for p in model.parameters():
        p.requires_grad = False
    
    # Create LoRA weights (these WILL have gradients)
    lora_A = torch.randn(64, 8, requires_grad=True)
    lora_B = torch.zeros(8, 128, requires_grad=True)  # Zero-init
    
    # Create injector
    injector = HookBasedLoRAInjector(model, scaling=1.0)
    
    # Test 1: Forward pass with LoRA
    x = torch.randn(2, 10, 64)
    
    with injector.apply_lora({"layer1": (lora_A, lora_B)}):
        output = model(x)
    
    print(f"✓ Forward pass completed. Output shape: {output.shape}")
    
    # Test 2: Gradient flow
    loss = output.sum()
    loss.backward()
    
    lora_A_has_grad = lora_A.grad is not None
    lora_B_has_grad = lora_B.grad is not None
    model_has_no_grad = all(p.grad is None for p in model.parameters())
    
    print(f"✓ LoRA A has gradient: {lora_A_has_grad}")
    print(f"✓ LoRA B has gradient: {lora_B_has_grad}")
    print(f"✓ Model params have no gradient: {model_has_no_grad}")
    
    # Test 3: Zero-init behavior
    lora_A_2 = torch.randn(64, 8)
    lora_B_2 = torch.zeros(8, 128)  # Zero B means no change
    
    with torch.no_grad():
        base_output = model(x)
        
    with torch.no_grad():
        with injector.apply_lora({"layer1": (lora_A_2, lora_B_2)}):
            lora_output = model(x)
    
    diff = (base_output - lora_output).abs().max().item()
    print(f"✓ Zero-init B preserves output: diff = {diff:.6f}")
    
    # Summary
    all_passed = lora_A_has_grad and lora_B_has_grad and model_has_no_grad and diff < 1e-5
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Hook-based injection works!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return all_passed


def test_layer_discovery():
    """Test automatic layer discovery."""
    print("\n" + "=" * 60)
    print("Testing Layer Discovery")
    print("=" * 60)
    
    # Create a mock Qwen-like structure
    class MockAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)
            
    class MockLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.self_attn = MockAttention(dim)
            
    class MockModel(nn.Module):
        def __init__(self, num_layers, dim):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer(dim) for _ in range(num_layers)])
    
    model = MockModel(num_layers=4, dim=64)
    
    # Discover layers
    targets = discover_target_layers(model)
    
    print(f"Discovered {len(targets)} target layers:")
    for t in targets[:8]:  # Show first 8
        print(f"  - {t}")
    if len(targets) > 8:
        print(f"  ... and {len(targets) - 8} more")
    
    expected = 4 * 4  # 4 layers × 4 projections
    passed = len(targets) == expected
    
    print(f"\n✓ Expected {expected} layers, found {len(targets)}: {'PASS' if passed else 'FAIL'}")
    
    return passed


if __name__ == "__main__":
    test_hook_based_injection()
    test_layer_discovery()
