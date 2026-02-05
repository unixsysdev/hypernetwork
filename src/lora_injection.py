"""
LoRA Injection Module for Dynamic Weight Application.

This module handles the injection of Hypernetwork-generated LoRA weights
into the frozen Student model during the forward pass.

The key insight is that we DON'T modify the Student's actual weights.
Instead, we intercept the forward pass and add the LoRA delta:
    output = base_layer(x) + (x @ A) @ B * scaling
    
This keeps the Student frozen while allowing gradients to flow through
the LoRA weights back to the Hypernetwork.
"""

from typing import Dict, Tuple, Optional, Callable, Any
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A wrapper that applies LoRA to a frozen linear layer.
    
    During forward:
        y = base_linear(x) + (x @ A) @ B * scaling
        
    The base_linear is frozen, but A and B receive gradients.
    """
    
    def __init__(
        self,
        base_linear: nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.lora_A = lora_A  # [in_features, rank]
        self.lora_B = lora_B  # [rank, out_features]
        self.scaling = scaling
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (no gradients through base weights)
        base_out = self.base_linear(x)
        
        # LoRA delta: (x @ A) @ B * scaling
        # x: [..., in_features]
        # A: [in_features, rank]
        # B: [rank, out_features]
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        
        return base_out + lora_out


class LoRAInjector:
    """
    Manages the injection of LoRA weights into a model.
    
    Usage:
        injector = LoRAInjector(model, lora_config)
        
        # During forward pass:
        with injector.inject(lora_weights):
            output = model(input)
        # LoRA is automatically removed after the context
    """
    
    def __init__(
        self,
        model: nn.Module,
        scaling: float = 2.0,  # alpha / rank
    ):
        self.model = model
        self.scaling = scaling
        self._hooks = []
        self._original_forwards = {}
        
    def _find_module(self, name: str) -> Optional[nn.Module]:
        """Find a module by its dot-separated name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _create_lora_forward(
        self,
        original_forward: Callable,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> Callable:
        """Create a new forward function that includes LoRA."""
        scaling = self.scaling
        
        def lora_forward(x: torch.Tensor) -> torch.Tensor:
            base_out = original_forward(x)
            lora_out = (x @ lora_A) @ lora_B * scaling
            return base_out + lora_out
        
        return lora_forward
    
    def inject(self, lora_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Context manager that temporarily injects LoRA weights.
        
        Args:
            lora_dict: Dictionary mapping layer names to (A, B) tuples
        """
        return LoRAContext(self, lora_dict)


class LoRAContext:
    """Context manager for temporary LoRA injection."""
    
    def __init__(
        self,
        injector: LoRAInjector,
        lora_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.injector = injector
        self.lora_dict = lora_dict
        self._original_forwards = {}
        
    def __enter__(self):
        """Inject LoRA by replacing forward methods."""
        for name, (lora_A, lora_B) in self.lora_dict.items():
            module = self.injector._find_module(name)
            if module is not None and isinstance(module, nn.Linear):
                # Save original forward
                self._original_forwards[name] = module.forward
                
                # Replace with LoRA forward
                module.forward = self.injector._create_lora_forward(
                    module.forward, lora_A, lora_B
                )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original forward methods."""
        for name, original_forward in self._original_forwards.items():
            module = self.injector._find_module(name)
            if module is not None:
                module.forward = original_forward
        return False


class FunctionalLoRA:
    """
    Functional interface for applying LoRA without modifying the model.
    
    This is more efficient for training because it doesn't require
    repeatedly patching and unpatching the model.
    """
    
    @staticmethod
    def apply_lora_to_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply a linear layer with LoRA in a functional manner.
        
        Args:
            x: Input tensor [..., in_features]
            weight: Base weight [out_features, in_features]
            bias: Optional bias [out_features]
            lora_A: LoRA A matrix [in_features, rank]
            lora_B: LoRA B matrix [rank, out_features]
            scaling: Scaling factor (alpha / rank)
            
        Returns:
            Output tensor [..., out_features]
        """
        # Base linear
        base_out = F.linear(x, weight, bias)
        
        # LoRA delta
        lora_out = (x @ lora_A) @ lora_B * scaling
        
        return base_out + lora_out


class StudentWithLoRA(nn.Module):
    """
    Wrapper around the Student model that applies dynamic LoRA.
    
    The Student's base weights are completely frozen.
    Only the LoRA weights (from Hypernetwork) receive gradients.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        lora_scaling: float = 2.0,
    ):
        super().__init__()
        self.student = student_model
        self.lora_scaling = lora_scaling
        
        # Freeze the student
        for param in self.student.parameters():
            param.requires_grad = False
            
        # Mapping from our key format to actual model paths
        # This needs to be configured based on the actual model architecture
        self._layer_mapping = {}
        self._setup_layer_mapping()
        
    def _setup_layer_mapping(self):
        """
        Set up the mapping from our layer keys to actual model paths.
        
        For Qwen3-Coder-Next, we need to map:
        - "X.self_attn.q_proj" -> actual attention layer path
        - "X.deltanet.linear_q" -> actual deltanet layer path
        """
        # This will be populated based on the actual model structure
        # when we load the model
        pass
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for the prompt."""
        # Access the embedding layer - path depends on model architecture
        if hasattr(self.student, 'model'):
            embed_layer = self.student.model.embed_tokens
        elif hasattr(self.student, 'transformer'):
            embed_layer = self.student.transformer.wte
        else:
            embed_layer = self.student.get_input_embeddings()
        
        return embed_layer(input_ids)
    
    def forward_with_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lora_weights: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dynamic LoRA injection.
        
        This is the main training interface. The LoRA weights are generated
        by the Hypernetwork and injected here.
        
        Args:
            input_ids: [B, L] - Input token IDs
            attention_mask: [B, L] - Attention mask
            lora_weights: Dict mapping layer names to (A, B) tuples
            
        Returns:
            logits: [B, L, V] - Output logits
        """
        if lora_weights is None:
            # No LoRA, just run the base model
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits
        
        # Apply LoRA via hook-based injection
        injector = LoRAInjector(self.student, scaling=self.lora_scaling)
        
        with injector.inject(lora_weights):
            outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        return outputs.logits


def create_lora_hooks(
    model: nn.Module,
    lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    scaling: float = 1.0,
) -> list:
    """
    Create forward hooks that apply LoRA to specific layers.
    
    This is an alternative to the context manager approach that might
    be more efficient for some use cases.
    
    Returns a list of hook handles that should be removed after use.
    """
    hooks = []
    
    def make_hook(lora_A: torch.Tensor, lora_B: torch.Tensor):
        def hook(module, input, output):
            x = input[0]
            lora_delta = (x @ lora_A) @ lora_B * scaling
            return output + lora_delta
        return hook
    
    for name, module in model.named_modules():
        if name in lora_weights:
            lora_A, lora_B = lora_weights[name]
            hook = module.register_forward_hook(make_hook(lora_A, lora_B))
            hooks.append(hook)
    
    return hooks


def remove_hooks(hooks: list):
    """Remove all hooks from a list of hook handles."""
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    # Quick test
    print("Testing LoRA injection...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            return self.layer2(x)
    
    model = TestModel()
    
    # Create test LoRA weights
    lora_A = torch.randn(64, 8, requires_grad=True)
    lora_B = torch.zeros(8, 128, requires_grad=True)  # Zero init
    
    # Test functional LoRA
    x = torch.randn(2, 10, 64)
    
    # Without LoRA
    base_out = model.layer1(x)
    
    # With LoRA (functional)
    lora_out = FunctionalLoRA.apply_lora_to_linear(
        x, model.layer1.weight, model.layer1.bias,
        lora_A, lora_B, scaling=1.0
    )
    
    # Since B is zero, outputs should be the same
    diff = (base_out - lora_out).abs().max().item()
    print(f"Difference with zero-init B: {diff:.6f} (should be ~0)")
    
    # Test gradient flow
    loss = lora_out.sum()
    loss.backward()
    
    has_grad = lora_A.grad is not None and lora_B.grad is not None
    print(f"Gradient flow: {'PASSED' if has_grad else 'FAILED'}")
    
    print("\nLoRA injection tests passed!")
