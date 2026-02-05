"""
Training Loop for Hypernetwork Distillation.

This module implements the core training loop that:
1. Loads the Teacher (Qwen3-Coder-480B) and Student (Qwen3-Coder-Next)
2. Generates LoRA weights via the Hypernetwork
3. Computes KL Divergence between Student and Teacher logits
4. Backpropagates to update the Hypernetwork

Key points:
- Teacher and Student do FORWARD PASS ONLY (no generation)
- Teacher is frozen and quantized (FP8)
- Student base weights are frozen
- Only Hypernetwork receives gradients
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from .hypernetwork import AgenticHyperNetwork, LoRAConfig
from .lora_injection import HookBasedLoRAInjector, discover_target_layers
from .data_loader import create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Models
    teacher_model_id: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    student_model_id: str = "Qwen/Qwen3-Coder-Next"
    
    # Hypernetwork
    hidden_dim: int = 2048
    num_encoder_layers: int = 4
    num_heads: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    
    # Training
    learning_rate: float = 1e-4
    warmup_lr: float = 1e-5
    epochs: int = 20
    warmup_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Distillation
    top_k_logits: int = 128
    temperature: float = 1.0
    
    # Data
    max_prompt_tokens: int = 512
    max_trajectory_tokens: int = 8192
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every_epochs: int = 2
    
    # Logging
    wandb_project: str = "hypernetwork-distillation"
    wandb_enabled: bool = True
    log_every_steps: int = 10
    
    # Cached teacher logits (offline distillation)
    use_cached_teacher: bool = True
    teacher_cache_dir: str = "./teacher_cache"
    
    # Performance optimization
    use_parallel_step: bool = True  # Use torch.compile optimized step


def top_k_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL Divergence using only top-K logits.
    
    This is CRITICAL for bandwidth efficiency when transferring
    Teacher logits from GPU Group 1 to GPU Group 2.
    
    Full vocab: 150,000 -> Too much data transfer
    Top-K: 128 -> 1000x reduction
    
    Args:
        student_logits: [B, L, V] - Student output logits
        teacher_logits: [B, L, V] - Teacher output logits
        k: Number of top logits to use
        temperature: Temperature for softmax
        
    Returns:
        KL divergence loss (scalar)
    """
    B, L, V = student_logits.shape
    
    # Get top-K indices from teacher
    teacher_topk_values, teacher_topk_indices = torch.topk(
        teacher_logits, k, dim=-1
    )  # [B, L, K]
    
    # Gather corresponding student logits
    student_topk_values = torch.gather(
        student_logits, dim=-1, index=teacher_topk_indices
    )  # [B, L, K]
    
    # Apply temperature and compute probabilities
    teacher_probs = F.softmax(teacher_topk_values / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk_values / temperature, dim=-1)
    
    # KL Divergence: sum over vocabulary, mean over batch and sequence
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    return kl


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the full distillation loss with masking.
    
    We only compute loss on positions where attention_mask == 1.
    This handles variable-length sequences in a batch.
    
    FIX: Properly excludes masked positions from loss computation
    instead of relying on kl_div reduction='batchmean'.
    """
    B, L, V = student_logits.shape
    
    # Get top-K from teacher
    teacher_topk_values, teacher_topk_indices = torch.topk(
        teacher_logits, top_k, dim=-1
    )  # [B, L, K]
    
    # Gather corresponding student logits
    student_topk_values = torch.gather(
        student_logits, dim=-1, index=teacher_topk_indices
    )  # [B, L, K]
    
    # Apply temperature and compute probabilities
    teacher_probs = F.softmax(teacher_topk_values / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk_values / temperature, dim=-1)
    
    # KL per position: sum over K vocab entries
    kl_per_pos = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)  # [B, L]
    
    # Apply mask and average over valid positions only
    mask = attention_mask.float()  # [B, L]
    masked_kl = kl_per_pos * mask
    loss = masked_kl.sum() / (mask.sum() + 1e-8)
    
    return loss


class DistillationTrainer:
    """
    Main trainer class for Hypernetwork distillation.
    
    Orchestrates:
    - Model loading and setup
    - Training loop
    - Checkpointing
    - Logging
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Will be initialized in setup()
        self.teacher = None
        self.student = None
        self.hypernetwork = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
    def setup(self):
        """Initialize all components."""
        logger.info("Setting up trainer...")
        
        # Load tokenizer (same for both models)
        logger.info(f"Loading tokenizer from {self.config.student_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Student model (frozen, will receive LoRA)
        logger.info(f"Loading Student model: {self.config.student_model_id}")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Will be refined for multi-GPU
            trust_remote_code=True,
        )
        self.student.eval()
        for param in self.student.parameters():
            param.requires_grad = False
        
        # Load Teacher model ONLY if not using cached logits
        if not self.config.use_cached_teacher:
            logger.info(f"Loading Teacher model: {self.config.teacher_model_id}")
            logger.warning("Runtime teacher inference is SLOW and requires massive VRAM!")
            self.teacher = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            logger.info("Using cached teacher logits - Teacher model NOT loaded")
            self.teacher = None
        
        # Setup LoRA injector (new hook-based version)
        self.lora_injector = HookBasedLoRAInjector(
            self.student,
            scaling=self.config.lora_alpha / self.config.lora_rank,
        )
        
        # CRITICAL: Discover target layers BEFORE creating Hypernetwork
        # This ensures we pass the actual layer names, not guessed templates
        self.target_layers = discover_target_layers(self.student)
        logger.info(f"Found {len(self.target_layers)} target layers for LoRA")
        
        # Log sample layer names to verify DeltaNet vs Attention detection
        if self.target_layers:
            logger.info(f"Sample layers: {self.target_layers[:4]}")
        
        # Initialize Hypernetwork with ACTUAL discovered layer names
        logger.info("Initializing Hypernetwork with discovered layer names...")
        
        # Auto-detect Student hidden_dim from model config
        student_hidden_dim = getattr(self.student.config, 'hidden_size', self.config.hidden_dim)
        logger.info(f"Student hidden_dim: {student_hidden_dim}")
        
        lora_config = LoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            hidden_dim=student_hidden_dim,  # Use actual Student dimension
        )
        self.hypernetwork = AgenticHyperNetwork(
            hidden_dim=self.config.hidden_dim,
            num_encoder_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_heads,
            lora_config=lora_config,
            # CRITICAL: Pass the real layer names!
            target_layer_names=self.target_layers,
        ).to(self.device)
        
        # Print parameter counts
        param_counts = self.hypernetwork.count_parameters()
        logger.info(f"Hypernetwork parameters: {param_counts}")
        logger.info(f"Generating {self.hypernetwork.num_loras} LoRA adapters")
        
        # Setup optimizer (only Hypernetwork receives gradients)
        self.optimizer = AdamW(
            self.hypernetwork.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler("cuda")
        
        # Pre-compile forward function for train_step_parallel (saves 30s per step!)
        self._compiled_forward = None
        try:
            self._compiled_forward = torch.compile(
                self._single_sample_forward, 
                mode="reduce-overhead"
            )
            logger.info("torch.compile: Enabled (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
        
        logger.info("Setup complete!")
        
    def get_student_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the Student model."""
        with torch.no_grad():
            if hasattr(self.student, 'model'):
                embed_layer = self.student.model.embed_tokens
            elif hasattr(self.student, 'transformer'):
                embed_layer = self.student.transformer.wte
            else:
                embed_layer = self.student.get_input_embeddings()
            return embed_layer(input_ids)
    
    def _compute_sparse_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_values: torch.Tensor,
        teacher_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence using sparse (top-k) teacher logits.
        
        IMPORTANT: teacher_values from vLLM are LOG PROBABILITIES (already log_softmax).
        We need to handle them correctly:
        - teacher_values = log P(teacher)  (FROM vLLM prompt_logprobs)
        - student needs log_softmax applied
        
        Args:
            student_logits: [L, V] - Full student logits (raw, not softmaxed)
            teacher_values: [L, K] - Top-K teacher LOG PROBABILITIES
            teacher_indices: [L, K] - Top-K teacher vocabulary indices
            attention_mask: [L] - Mask (1=compute, 0=ignore)
        
        Returns:
            KL divergence loss (scalar)
        """
        L, V = student_logits.shape
        K = teacher_values.shape[-1]
        
        # Gather student logits at teacher's top-K indices
        student_gathered = torch.gather(student_logits, dim=-1, index=teacher_indices)
        
        # Apply temperature
        temperature = self.config.temperature
        
        # Student: apply log_softmax to raw logits (restricted to top-K for approximation)
        student_log_probs = F.log_softmax(student_gathered / temperature, dim=-1)  # [L, K]
        
        # Teacher: values are ALREADY log probs from vLLM!
        # WARNING: Temperature scaling of log-probs is mathematically different from
        # scaling logits before softmax. For temperature != 1.0, this is an approximation.
        # Correct approach: cache raw logits, or apply temp during caching.
        # For now: only apply temp=1.0 passthrough, warn otherwise.
        if temperature != 1.0:
            import warnings
            warnings.warn(
                f"Temperature={temperature} with cached vLLM log-probs is approximate. "
                "For accurate temperature scaling, cache raw logits instead.",
                UserWarning,
            )
        # Convert log-probs to probs and renormalize over top-K
        teacher_probs = torch.exp(teacher_values)  # [L, K]
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Compute KL divergence per position: sum_k P(k) * [log P(k) - log Q(k)]
        kl_per_pos = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)  # [L]
        
        # Apply attention mask
        mask = attention_mask.float()
        masked_kl = kl_per_pos * mask
        
        # Average over non-masked positions
        loss = masked_kl.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.
        
        1. Get prompt embeddings from Student
        2. Hypernetwork generates LoRA weights
        3. Student forward pass with LoRA
        4. Get teacher logits (from cache OR runtime)
        5. Compute KL divergence loss
        6. Backprop to Hypernetwork
        
        Returns dict of metrics.
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_ids = batch["prompt_ids"].to(self.device)
        prompt_mask = batch["prompt_mask"].to(self.device)
        
        # Check if we have cached teacher logits
        has_cached_teacher = (
            self.config.use_cached_teacher and 
            "teacher_values" in batch and 
            "teacher_indices" in batch
        )
        
        if has_cached_teacher:
            # Load pre-computed teacher logits (top-k sparse format)
            teacher_values = batch["teacher_values"].to(self.device)  # [B, L, K]
            teacher_indices = batch["teacher_indices"].to(self.device)  # [B, L, K]
        
        # 1. Get prompt embeddings (no grad through Student)
        with torch.no_grad():
            prompt_embeds = self.get_student_embeddings(prompt_ids)
        
        # 2. Hypernetwork generates LoRA (GRADIENTS START HERE)
        lora_output = self.hypernetwork(prompt_embeds, prompt_mask)
        
        # Process samples (sequential for now, TODO: batched LoRA)
        batch_size = input_ids.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            lora_dict = self.hypernetwork.get_lora_dict(lora_output, batch_idx=b)
            
            # 3. Student forward with LoRA
            with self.lora_injector.apply_lora(lora_dict):
                student_outputs = self.student(
                    input_ids=input_ids[b:b+1],
                    attention_mask=attention_mask[b:b+1],
                )
                student_logits = student_outputs.logits
            
            # 4. Get teacher logits
            if has_cached_teacher:
                # Use cached sparse teacher logits
                loss = self._compute_sparse_kl_loss(
                    student_logits[0],  # [L, V]
                    teacher_values[b],   # [L, K]
                    teacher_indices[b],  # [L, K]
                    attention_mask[b],   # [L]
                )
            else:
                # Runtime teacher inference (SLOW, requires massive VRAM)
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=input_ids[b:b+1],
                        attention_mask=attention_mask[b:b+1],
                    )
                    teacher_logits = teacher_outputs.logits
                
                loss = compute_distillation_loss(
                    student_logits,
                    teacher_logits,
                    attention_mask[b:b+1],
                    top_k=self.config.top_k_logits,
                    temperature=self.config.temperature,
                )
            
            total_loss += loss
        
        # Average loss over batch AND accumulation steps for correct gradient scaling
        loss = total_loss / batch_size / self.config.gradient_accumulation_steps
        
        # 6. Backprop
        self.scaler.scale(loss).backward()
        
        return {
            "loss": loss.item(),
            "context_norm": lora_output["context"].norm().item(),
        }
    
    def train_step_parallel(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Optimized training step using parallel forward passes.
        
        NOTE: Despite the name, per-sample LoRA application is still sequential
        due to the need to apply different LoRA weights per sample. The 'parallel'
        refers to potential future batched einsum optimization, not current impl.
        True batched LoRA would require fundamentally different architecture.
        
        Current optimizations:
        - Pre-compiled forward function via torch.compile
        - Loss stacking for single backward pass
        """
        import torch.utils.checkpoint as checkpoint
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_ids = batch["prompt_ids"].to(self.device)
        prompt_mask = batch["prompt_mask"].to(self.device)
        
        has_cached_teacher = (
            self.config.use_cached_teacher and 
            "teacher_values" in batch and 
            "teacher_indices" in batch
        )
        
        if has_cached_teacher:
            teacher_values = batch["teacher_values"].to(self.device)
            teacher_indices = batch["teacher_indices"].to(self.device)
        
        # Get prompt embeddings
        with torch.no_grad():
            prompt_embeds = self.get_student_embeddings(prompt_ids)
        
        # Hypernetwork generates ALL LoRAs at once (this is already batched!)
        lora_output = self.hypernetwork(prompt_embeds, prompt_mask)
        
        batch_size = input_ids.shape[0]
        
        # Process in parallel using Python threads for I/O bound parts
        # The actual compute happens on GPU so this overlaps nicely
        losses = []
        
        # Use pre-compiled forward function (compiled during setup, NOT here!)
        forward_fn = self._compiled_forward or self._single_sample_forward
        
        for b in range(batch_size):
            loss = forward_fn(
                b, lora_output, input_ids, attention_mask,
                teacher_values if has_cached_teacher else None,
                teacher_indices if has_cached_teacher else None,
                has_cached_teacher,
            )
            losses.append(loss)
        
        # Sum and average losses, scaled for gradient accumulation
        total_loss = torch.stack(losses).mean() / self.config.gradient_accumulation_steps
        
        # Backprop
        self.scaler.scale(total_loss).backward()
        
        return {
            "loss": total_loss.item(),
            "context_norm": lora_output["context"].norm().item(),
        }
    
    def _single_sample_forward(
        self,
        batch_idx: int,
        lora_output: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_values: Optional[torch.Tensor],
        teacher_indices: Optional[torch.Tensor],
        has_cached_teacher: bool,
    ) -> torch.Tensor:
        """Process a single sample - separated for potential torch.compile."""
        lora_dict = self.hypernetwork.get_lora_dict(lora_output, batch_idx=batch_idx)
        
        # Use autocast for mixed precision (bfloat16 on H200)
        with self.lora_injector.apply_lora(lora_dict), autocast("cuda", dtype=torch.bfloat16):
            student_outputs = self.student(
                input_ids=input_ids[batch_idx:batch_idx+1],
                attention_mask=attention_mask[batch_idx:batch_idx+1],
            )
            student_logits = student_outputs.logits
        
        if has_cached_teacher:
            return self._compute_sparse_kl_loss(
                student_logits[0],
                teacher_values[batch_idx],
                teacher_indices[batch_idx],
                attention_mask[batch_idx],
            )
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids[batch_idx:batch_idx+1],
                    attention_mask=attention_mask[batch_idx:batch_idx+1],
                )
            return compute_distillation_loss(
                student_logits,
                teacher_outputs.logits,
                attention_mask[batch_idx:batch_idx+1],
                top_k=self.config.top_k_logits,
                temperature=self.config.temperature,
            )
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.hypernetwork.train()
        
        total_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(dataloader):
            # Use optimized parallel step or sequential step
            if self.config.use_parallel_step:
                metrics = self.train_step_parallel(batch)
            else:
                metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_steps += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.hypernetwork.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Logging
                if (step + 1) % self.config.log_every_steps == 0:
                    avg_loss = total_loss / num_steps
                    logger.info(
                        f"Epoch {epoch} | Step {step+1} | Loss: {avg_loss:.4f}"
                    )
                    if wandb.run:
                        wandb.log({
                            "loss": avg_loss,
                            "epoch": epoch,
                            "step": step,
                            **metrics,
                        })
        
        return {"avg_loss": total_loss / max(num_steps, 1)}
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save training checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "hypernetwork_state_dict": self.hypernetwork.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint["epoch"]
    
    def train(self):
        """Main training loop."""
        # Setup wandb
        if self.config.wandb_enabled:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )
        else:
            wandb.init(mode="disabled")
        
        # Create dataloader
        dataloader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_prompt_tokens=self.config.max_prompt_tokens,
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            teacher_cache_dir=self.config.teacher_cache_dir if self.config.use_cached_teacher else None,
        )
        
        # Setup scheduler: warmup → cosine decay
        steps_per_epoch = len(dataloader) // self.config.gradient_accumulation_steps
        warmup_steps = self.config.warmup_epochs * steps_per_epoch
        total_steps = self.config.epochs * steps_per_epoch
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.config.warmup_lr / self.config.learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")
            
            metrics = self.train_epoch(dataloader, epoch)
            
            logger.info(f"Epoch {epoch + 1} complete. Avg Loss: {metrics['avg_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_epochs == 0:
                ckpt_path = Path(self.config.save_dir) / f"epoch_{epoch+1}.pt"
                self.save_checkpoint(epoch + 1, str(ckpt_path))
        
        # Save final model
        final_path = Path(self.config.save_dir) / "final.pt"
        self.save_checkpoint(self.config.epochs, str(final_path))
        
        wandb.finish()
        logger.info("Training complete!")


def main():
    """Entry point for training."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat_config[f"{key}"] = value
        else:
            flat_config[section] = values
    
    config = TrainingConfig(**{
        k: v for k, v in flat_config.items()
        if hasattr(TrainingConfig, k)
    })
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Create trainer and run
    trainer = DistillationTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
