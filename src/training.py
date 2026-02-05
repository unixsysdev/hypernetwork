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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from .hypernetwork import AgenticHyperNetwork, LoRAConfig
from .lora_injection import LoRAInjector
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
    log_every_steps: int = 10


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
    """
    # Mask out padding positions
    # attention_mask: [B, L]
    # Expand to [B, L, 1] for masking logits
    mask = attention_mask.unsqueeze(-1).float()
    
    # Apply mask (set masked positions to very negative value)
    large_neg = -1e9
    student_logits = student_logits * mask + large_neg * (1 - mask)
    teacher_logits = teacher_logits * mask + large_neg * (1 - mask)
    
    # Compute top-K KL divergence
    loss = top_k_kl_divergence(
        student_logits,
        teacher_logits,
        k=top_k,
        temperature=temperature,
    )
    
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
        
        # Load Teacher model (frozen, quantized)
        logger.info(f"Loading Teacher model: {self.config.teacher_model_id}")
        # Note: For actual 8xH200 setup, you'd use vLLM here
        # This is the transformers fallback for smaller setups
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            # For FP8 quantization, use:
            # load_in_8bit=True,  # or quantization_config for FP8
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Initialize Hypernetwork
        logger.info("Initializing Hypernetwork...")
        lora_config = LoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        )
        self.hypernetwork = AgenticHyperNetwork(
            hidden_dim=self.config.hidden_dim,
            num_encoder_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_heads,
            lora_config=lora_config,
        ).to(self.device)
        
        # Print parameter counts
        param_counts = self.hypernetwork.count_parameters()
        logger.info(f"Hypernetwork parameters: {param_counts}")
        
        # Setup optimizer (only Hypernetwork receives gradients)
        self.optimizer = AdamW(
            self.hypernetwork.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler()
        
        # Setup LoRA injector
        self.lora_injector = LoRAInjector(
            self.student,
            scaling=self.config.lora_alpha / self.config.lora_rank,
        )
        
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
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.
        
        1. Get prompt embeddings from Student
        2. Hypernetwork generates LoRA weights
        3. Student forward pass with LoRA
        4. Teacher forward pass (scoring only)
        5. Compute KL divergence loss
        6. Backprop to Hypernetwork
        
        Returns dict of metrics.
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_ids = batch["prompt_ids"].to(self.device)
        prompt_mask = batch["prompt_mask"].to(self.device)
        
        # 1. Get prompt embeddings (no grad through Student)
        with torch.no_grad():
            prompt_embeds = self.get_student_embeddings(prompt_ids)
        
        # 2. Hypernetwork generates LoRA (GRADIENTS START HERE)
        lora_output = self.hypernetwork(prompt_embeds, prompt_mask)
        
        # Convert to dict format for injection
        # For batched training, we process one sample at a time
        # TODO: Optimize for batched LoRA injection
        batch_size = input_ids.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            lora_dict = self.hypernetwork.get_lora_dict(lora_output, batch_idx=b)
            
            # 3. Student forward with LoRA
            with self.lora_injector.inject(lora_dict):
                student_outputs = self.student(
                    input_ids=input_ids[b:b+1],
                    attention_mask=attention_mask[b:b+1],
                )
                student_logits = student_outputs.logits
            
            # 4. Teacher forward (no grad, just scoring)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids[b:b+1],
                    attention_mask=attention_mask[b:b+1],
                )
                teacher_logits = teacher_outputs.logits
            
            # 5. Compute loss
            loss = compute_distillation_loss(
                student_logits,
                teacher_logits,
                attention_mask[b:b+1],
                top_k=self.config.top_k_logits,
                temperature=self.config.temperature,
            )
            total_loss += loss
        
        # Average loss over batch
        loss = total_loss / batch_size
        
        # 6. Backprop
        self.scaler.scale(loss).backward()
        
        return {
            "loss": loss.item(),
            "context_norm": lora_output["context"].norm().item(),
        }
    
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
            # Accumulate gradients
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
        wandb.init(
            project=self.config.wandb_project,
            config=vars(self.config),
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_prompt_tokens=self.config.max_prompt_tokens,
            max_trajectory_tokens=self.config.max_trajectory_tokens,
        )
        
        # Setup scheduler
        total_steps = len(dataloader) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Warmup phase uses lower LR
            if epoch < self.config.warmup_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.config.warmup_lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.config.learning_rate
            
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
