"""
Test: Gradient Flow Verification

CRITICAL: Run this test BEFORE starting full training.

This test verifies that gradients flow correctly from the loss
all the way back to the Hypernetwork. If this test fails,
your training will silently do nothing.

The test checks:
1. The computational graph connects Loss -> Student -> LoRA -> Hypernetwork
2. Gradients are non-zero (not vanishing)
3. The gradient magnitude is reasonable (not exploding)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.hypernetwork import AgenticHyperNetwork, LoRAConfig
from src.lora_injection import FunctionalLoRA


def test_gradient_flow_basic():
    """
    Basic test: Verify gradients reach the Hypernetwork.
    
    Uses a simplified model to isolate the gradient flow mechanism.
    """
    print("=" * 60)
    print("TEST: Basic Gradient Flow")
    print("=" * 60)
    
    # Create hypernetwork
    config = LoRAConfig(rank=8, alpha=16)
    hypernet = AgenticHyperNetwork(
        hidden_dim=256,  # Smaller for testing
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Create mock prompt embeddings
    batch_size = 2
    seq_len = 64
    hidden_dim = 256
    
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    # Forward through hypernetwork
    output = hypernet(prompt_embeds, prompt_mask)
    
    # Simulate a loss using the LoRA output
    attn_A, attn_B = output["attention"]
    delta_A, delta_B = output["deltanet"]
    
    # Simple loss: sum of all LoRA parameters
    loss = attn_A.sum() + attn_B.sum() + delta_A.sum() + delta_B.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    checks = []
    
    # Check prompt encoder
    if hypernet.prompt_encoder.pool_query.grad is not None:
        grad_norm = hypernet.prompt_encoder.pool_query.grad.norm().item()
        checks.append(("Prompt Encoder", grad_norm > 0, grad_norm))
    else:
        checks.append(("Prompt Encoder", False, 0.0))
    
    # Check attention generator
    for name, param in hypernet.attention_generator.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            checks.append((f"Attention Generator: {name}", grad_norm > 0, grad_norm))
            break  # Just check one
    
    # Check deltanet generator
    for name, param in hypernet.deltanet_generator.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            checks.append((f"DeltaNet Generator: {name}", grad_norm > 0, grad_norm))
            break
    
    # Print results
    all_passed = True
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name} | grad_norm = {value:.6f}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("RESULT: Basic gradient flow test PASSED")
    else:
        print("RESULT: Basic gradient flow test FAILED")
        print("ERROR: Gradients are not reaching the Hypernetwork!")
    
    return all_passed


def test_gradient_flow_with_lora():
    """
    Full test: Verify gradients flow through LoRA application.
    
    This simulates the actual training setup where:
    1. Hypernetwork generates LoRA
    2. LoRA is applied to a frozen model layer
    3. Loss is computed on the output
    4. Gradients flow back through LoRA to Hypernetwork
    """
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow Through LoRA Application")
    print("=" * 60)
    
    # Create hypernetwork
    hidden_dim = 256
    config = LoRAConfig(rank=8, alpha=16)
    hypernet = AgenticHyperNetwork(
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Create a "frozen" base layer (simulating Student layer)
    base_layer = nn.Linear(hidden_dim, hidden_dim)
    for param in base_layer.parameters():
        param.requires_grad = False  # FROZEN
    
    # Input data
    batch_size = 2
    seq_len = 64
    
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    prompt_mask = torch.ones(batch_size, seq_len)
    input_x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Generate LoRA
    output = hypernet(prompt_embeds, prompt_mask)
    
    # Get one LoRA matrix pair (for testing)
    attn_A, attn_B = output["attention"]
    lora_A = attn_A[0, 0]  # [hidden_dim, rank]
    lora_B = attn_B[0, 0]  # [rank, hidden_dim]
    
    # Apply LoRA to the base layer output
    base_out = base_layer(input_x)  # No gradients here
    lora_out = (input_x @ lora_A) @ lora_B * (config.alpha / config.rank)
    final_out = base_out + lora_out
    
    # Loss
    target = torch.randn_like(final_out)
    loss = F.mse_loss(final_out, target)
    
    # Backward
    loss.backward()
    
    # Verify gradients
    checks = []
    
    # Base layer should have NO gradients (frozen)
    base_has_grad = base_layer.weight.grad is not None
    checks.append(("Base Layer (should be None)", not base_has_grad, 0.0))
    
    # Hypernetwork should have gradients
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                checks.append((f"Hypernetwork: {name}", True, grad_norm))
                break  # Just need one to verify
    
    # Check if any gradients reached hypernet
    total_grad_norm = sum(
        p.grad.norm().item() 
        for p in hypernet.parameters() 
        if p.grad is not None
    )
    checks.append(("Total Hypernetwork Gradient", total_grad_norm > 0, total_grad_norm))
    
    # Print results
    all_passed = True
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name} | value = {value:.6f}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("RESULT: LoRA gradient flow test PASSED")
    else:
        print("RESULT: LoRA gradient flow test FAILED")
        print("ERROR: The gradient highway is broken!")
    
    return all_passed


def test_zero_initialization():
    """
    Test: Verify zero-initialization of LoRA B matrices.
    
    At step 0, the LoRA should have (nearly) no effect:
    output = base(x) + 0 = base(x)
    
    This ensures the Student isn't "lobotomized" at the start of training.
    """
    print("\n" + "=" * 60)
    print("TEST: Zero Initialization")
    print("=" * 60)
    
    # Create hypernetwork
    hidden_dim = 256
    config = LoRAConfig(rank=8, alpha=16)
    hypernet = AgenticHyperNetwork(
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Generate LoRA at initialization
    batch_size = 2
    seq_len = 64
    
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        output = hypernet(prompt_embeds, prompt_mask)
    
    # Check B matrices are near zero
    attn_A, attn_B = output["attention"]
    delta_A, delta_B = output["deltanet"]
    
    attn_B_norm = attn_B.abs().mean().item()
    delta_B_norm = delta_B.abs().mean().item()
    
    # Threshold: should be very close to zero
    threshold = 0.01
    
    checks = [
        ("Attention B matrix", attn_B_norm < threshold, attn_B_norm),
        ("DeltaNet B matrix", delta_B_norm < threshold, delta_B_norm),
    ]
    
    # Print results
    all_passed = True
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name} | mean_abs = {value:.6f} (threshold: {threshold})")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("RESULT: Zero initialization test PASSED")
        print("The Student will start with its original behavior.")
    else:
        print("RESULT: Zero initialization test FAILED")
        print("WARNING: LoRA may damage the Student at step 0!")
    
    return all_passed


def test_gradient_magnitude():
    """
    Test: Check gradient magnitudes are reasonable.
    
    - Too small: Vanishing gradients, no learning
    - Too large: Exploding gradients, unstable training
    """
    print("\n" + "=" * 60)
    print("TEST: Gradient Magnitude Check")
    print("=" * 60)
    
    # Create hypernetwork
    hidden_dim = 256
    config = LoRAConfig(rank=8, alpha=16)
    hypernet = AgenticHyperNetwork(
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_heads=4,
        lora_config=config,
    )
    
    # Simulate a realistic loss magnitude
    batch_size = 2
    seq_len = 64
    
    prompt_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    prompt_mask = torch.ones(batch_size, seq_len)
    
    output = hypernet(prompt_embeds, prompt_mask)
    
    # Simulate KL divergence-like loss (typical magnitude ~0.1 to 10)
    attn_A, attn_B = output["attention"]
    loss = F.mse_loss(attn_A, torch.zeros_like(attn_A))  # ~1.0 magnitude
    
    loss.backward()
    
    # Collect gradient statistics
    grad_norms = []
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    # Check bounds
    min_grad = 1e-8
    max_grad = 1e3
    
    all_passed = True
    for name, norm in grad_norms[:5]:  # Show first 5
        passed = min_grad < norm < max_grad
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name[:40]:40s} | grad_norm = {norm:.2e}")
        all_passed = all_passed and passed
    
    if len(grad_norms) > 5:
        print(f"  ... and {len(grad_norms) - 5} more parameters")
    
    # Summary statistics
    all_norms = [n for _, n in grad_norms]
    print(f"\n  Min grad norm: {min(all_norms):.2e}")
    print(f"  Max grad norm: {max(all_norms):.2e}")
    print(f"  Mean grad norm: {sum(all_norms)/len(all_norms):.2e}")
    
    print()
    if all_passed:
        print("RESULT: Gradient magnitude test PASSED")
    else:
        print("RESULT: Gradient magnitude test FAILED")
    
    return all_passed


def run_all_tests():
    """Run all gradient flow tests."""
    print("\n" + "#" * 60)
    print("# HYPERNETWORK GRADIENT FLOW VERIFICATION")
    print("#" * 60)
    print("\nRunning pre-flight checks before training...")
    print("If any test FAILS, DO NOT proceed with training.\n")
    
    results = []
    
    results.append(("Basic Gradient Flow", test_gradient_flow_basic()))
    results.append(("LoRA Application Flow", test_gradient_flow_with_lora()))
    results.append(("Zero Initialization", test_zero_initialization()))
    results.append(("Gradient Magnitude", test_gradient_magnitude()))
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for training!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before training!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
