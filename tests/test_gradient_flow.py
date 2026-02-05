#!/usr/bin/env python3
"""
Pre-training Verification Tests.

Run these tests BEFORE starting full training to verify:
1. Gradients flow correctly to the Hypernetwork
2. Zero-initialization preserves Student behavior
3. Gradient magnitudes are reasonable

Usage:
    python tests/test_gradient_flow.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_hypernetwork_gradient_flow():
    """Test that gradients reach the Hypernetwork components."""
    print("=" * 60)
    print("Test 1: Hypernetwork Gradient Flow")
    print("=" * 60)
    
    from src.hypernetwork import AgenticHyperNetwork, LoRAConfig
    
    # Create small config for testing
    config = LoRAConfig(rank=8, alpha=16, num_layers=4, hidden_dim=128)
    hypernet = AgenticHyperNetwork(
        hidden_dim=128,
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Simulate prompt embeddings
    batch_size = 2
    seq_len = 32
    prompt_embeds = torch.randn(batch_size, seq_len, 128, requires_grad=True)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output = hypernet(prompt_embeds, prompt_mask)
    
    # Compute dummy loss
    loss = output["lora_A"].sum() + output["lora_B"].sum()
    loss.backward()
    
    # Check gradients
    encoder_has_grad = hypernet.prompt_encoder.pool_query.grad is not None
    generator_has_grad = hypernet.lora_generator.layer_embed.weight.grad is not None
    
    print(f"  Prompt Encoder has gradients: {encoder_has_grad}")
    print(f"  LoRA Generator has gradients: {generator_has_grad}")
    
    passed = encoder_has_grad and generator_has_grad
    print(f"\n  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_lora_injection_gradient_flow():
    """Test that gradients flow through LoRA injection to Hypernetwork."""
    print("\n" + "=" * 60)
    print("Test 2: LoRA Injection Gradient Flow")
    print("=" * 60)
    
    from src.lora_injection import HookBasedLoRAInjector
    
    # Create mock "student" model
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            return self.layer2(x)
    
    student = MockStudent()
    
    # Freeze student
    for p in student.parameters():
        p.requires_grad = False
    
    # Create LoRA weights that WILL have gradients
    lora_A = torch.randn(64, 8, requires_grad=True)
    lora_B = torch.randn(8, 128, requires_grad=True) * 0.01  # Small for stability
    
    # Setup injector
    injector = HookBasedLoRAInjector(student, scaling=1.0)
    
    # Forward with LoRA
    x = torch.randn(2, 10, 64)
    
    with injector.apply_lora({"layer1": (lora_A, lora_B)}):
        output = student(x)
    
    # Backprop
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    lora_has_grad = lora_A.grad is not None and lora_B.grad is not None
    student_no_grad = all(p.grad is None for p in student.parameters())
    
    print(f"  LoRA weights have gradients: {lora_has_grad}")
    print(f"  Student weights have NO gradients: {student_no_grad}")
    
    passed = lora_has_grad and student_no_grad
    print(f"\n  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_zero_initialization():
    """Test that zero-init preserves Student behavior at step 0."""
    print("\n" + "=" * 60)
    print("Test 3: Zero Initialization")
    print("=" * 60)
    
    from src.hypernetwork import AgenticHyperNetwork, LoRAConfig
    from src.lora_injection import HookBasedLoRAInjector
    
    # Create models
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            
        def forward(self, x):
            return self.q_proj(x) + self.k_proj(x)
    
    student = MockStudent()
    
    config = LoRAConfig(rank=8, alpha=16, num_layers=2, hidden_dim=64)
    hypernet = AgenticHyperNetwork(
        hidden_dim=64,
        num_encoder_layers=1,
        num_heads=2,
        lora_config=config,
        layer_path_template="{module}",  # Simple naming for mock
    )
    
    injector = HookBasedLoRAInjector(student, scaling=2.0)
    
    # Get outputs with and without LoRA
    x = torch.randn(1, 5, 64)
    prompt_embeds = torch.randn(1, 10, 64)
    prompt_mask = torch.ones(1, 10)
    
    with torch.no_grad():
        # Output without LoRA
        base_output = student(x)
        
        # Generate LoRA from Hypernetwork
        lora_output = hypernet(prompt_embeds, prompt_mask)
        lora_dict = hypernet.get_lora_dict(lora_output, batch_idx=0)
        
        # Output with LoRA
        with injector.apply_lora(lora_dict):
            lora_applied_output = student(x)
    
    # Check difference
    diff = (base_output - lora_applied_output).abs().max().item()
    
    print(f"  Max difference between base and LoRA output: {diff:.6f}")
    print(f"  LoRA B matrix mean abs: {lora_output['lora_B'].abs().mean().item():.6f}")
    
    # With zero-init, B matrices should be ~0, so output should be nearly identical
    passed = diff < 0.1  # Allow small numerical differences
    print(f"\n  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_gradient_magnitude():
    """Test that gradients have reasonable magnitudes (not exploding/vanishing)."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Magnitude")
    print("=" * 60)
    
    from src.hypernetwork import AgenticHyperNetwork, LoRAConfig
    
    config = LoRAConfig(rank=8, alpha=16, num_layers=4, hidden_dim=128)
    hypernet = AgenticHyperNetwork(
        hidden_dim=128,
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Forward pass
    prompt_embeds = torch.randn(2, 32, 128)
    prompt_mask = torch.ones(2, 32)
    
    output = hypernet(prompt_embeds, prompt_mask)
    
    # Simulate distillation loss
    fake_target = torch.randn_like(output["lora_A"])
    loss = F.mse_loss(output["lora_A"], fake_target)
    loss.backward()
    
    # Check gradient norms
    grad_norms = []
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    avg_grad = sum(grad_norms) / len(grad_norms)
    max_grad = max(grad_norms)
    min_grad = min(grad_norms)
    
    print(f"  Gradient norm - Min: {min_grad:.6f}, Max: {max_grad:.6f}, Avg: {avg_grad:.6f}")
    
    # Good range: between 1e-6 and 1e3
    not_vanishing = min_grad > 1e-10
    not_exploding = max_grad < 1e5
    
    print(f"  Not vanishing (min > 1e-10): {not_vanishing}")
    print(f"  Not exploding (max < 1e5): {not_exploding}")
    
    passed = not_vanishing and not_exploding
    print(f"\n  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_layer_discovery():
    """Test automatic layer discovery in a model."""
    print("\n" + "=" * 60)
    print("Test 5: Layer Discovery")
    print("=" * 60)
    
    from src.lora_injection import discover_target_layers
    
    # Create mock model with Qwen-like structure
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
    
    targets = discover_target_layers(model)
    
    print(f"  Discovered {len(targets)} target layers")
    for t in targets[:4]:
        print(f"    - {t}")
    if len(targets) > 4:
        print(f"    ... and {len(targets) - 4} more")
    
    expected = 4 * 4  # 4 layers × 4 projections
    passed = len(targets) == expected
    print(f"\n  Expected {expected}, found {len(targets)}")
    print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HYPERNETWORK PRE-TRAINING VERIFICATION TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("Hypernetwork Gradient Flow", test_hypernetwork_gradient_flow()))
    results.append(("LoRA Injection Gradient Flow", test_lora_injection_gradient_flow()))
    results.append(("Zero Initialization", test_zero_initialization()))
    results.append(("Gradient Magnitude", test_gradient_magnitude()))
    results.append(("Layer Discovery", test_layer_discovery()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for training!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before training!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
