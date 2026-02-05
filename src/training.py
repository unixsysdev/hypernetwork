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
from torch.amp import autocast  # GradScaler not needed for bfloat16
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
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    warmup_lr: float = 1e-5
    epochs: int = 20
    warmup_epochs: int = 2
    batch_size: int = 4  # Batched LoRA via einsum enables true batching
    gradient_accumulation_steps: int = 2  # Effective batch size = 8
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
    wandb_run_name: str = ""
    wandb_enabled: bool = True
    log_every_steps: int = 10
    
    # Cached teacher logits (offline distillation)
    use_cached_teacher: bool = True
    teacher_cache_dir: str = "./teacher_cache"
    
    # Validation
    val_split: float = 0.05  # Hold out 5% for validation
    val_every_epochs: int = 1  # Evaluate every N epochs



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
    
    # CRITICAL: Apply log_softmax over FULL vocab BEFORE gathering!
    # This ensures the normalization constant is correct.
    student_log_probs_full = F.log_softmax(student_logits / temperature, dim=-1)  # [B, L, V]
    student_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_topk_indices)  # [B, L, K]
    
    # Teacher: softmax over just top-K (this is an approximation, but for teacher it's OK)
    teacher_probs = F.softmax(teacher_topk_values / temperature, dim=-1)  # [B, L, K]
    
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
        # Note: GradScaler is NOT needed for bfloat16 (same exponent range as fp32)
        # Note: torch.compile is NOT used because dynamic hook registration/removal
        # in apply_lora is incompatible with graph tracing (stale CUDA graphs).
        
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
        teacher_format: str = "logprobs",
    ) -> torch.Tensor:
        """
        Compute KL divergence using sparse (top-k) teacher logits.
        
        Handles both formats:
        - 'logprobs': teacher_values are log-probs from vLLM (exp to get probs)
        - 'logits': teacher_values are raw logits (apply softmax)
        
        Args:
            student_logits: [L, V] - Full student logits (raw, not softmaxed)
            teacher_values: [L, K] - Top-K teacher values (format-dependent)
            teacher_indices: [L, K] - Top-K teacher vocabulary indices
            attention_mask: [L] - Mask (1=compute, 0=ignore)
            teacher_format: 'logprobs' or 'logits'
        
        Returns:
            KL divergence loss (scalar)
        """
        L, V = student_logits.shape
        K = teacher_values.shape[-1]
        
        # Apply temperature
        temperature = self.config.temperature
        
        # CRITICAL: Apply log_softmax over FULL vocab BEFORE gathering!
        # This ensures the normalization constant is correct.
        # Previous bug: log_softmax(gathered) uses wrong normalization over K values
        student_log_probs_full = F.log_softmax(student_logits / temperature, dim=-1)  # [L, V]
        student_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_indices)  # [L, K]
        
        # Teacher: convert to probabilities based on format
        if teacher_format == "logits":
            # Raw logits: apply softmax with temperature
            teacher_probs = F.softmax(teacher_values / temperature, dim=-1)  # [L, K]
        else:
            # Log-probs from vLLM: exponentiate and renormalize
            # WARNING: Temperature scaling of log-probs is mathematically approximate
            if temperature != 1.0:
                import warnings
                warnings.warn(
                    f"Temperature={temperature} with cached vLLM log-probs is approximate. "
                    "For accurate temperature scaling, cache raw logits instead.",
                    UserWarning,
                )
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

    def _compute_batched_sparse_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_values: torch.Tensor,
        teacher_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_format: str = "logprobs",
    ) -> torch.Tensor:
        """
        Compute KL divergence for a full batch at once.
        
        Args:
            student_logits: [B, L, V] - Full student logits
            teacher_values: [B, L, K] - Top-K teacher values
            teacher_indices: [B, L, K] - Top-K teacher vocabulary indices
            attention_mask: [B, L] - Mask (1=compute, 0=ignore)
            teacher_format: 'logprobs' or 'logits'
        
        Returns:
            KL divergence loss (scalar, averaged over all valid positions in batch)
        """
        temperature = self.config.temperature
        
        # Full-vocab log_softmax, then gather at teacher indices
        student_log_probs_full = F.log_softmax(student_logits / temperature, dim=-1)  # [B, L, V]
        student_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_indices)  # [B, L, K]
        
        # Teacher probabilities
        if teacher_format == "logits":
            teacher_probs = F.softmax(teacher_values / temperature, dim=-1)  # [B, L, K]
        else:
            if temperature != 1.0:
                import warnings
                warnings.warn(
                    f"Temperature={temperature} with cached vLLM log-probs is approximate. "
                    "For accurate temperature scaling, cache raw logits instead.",
                    UserWarning,
                )
            teacher_probs = torch.exp(teacher_values)  # [B, L, K]
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
        
        # KL per position
        kl_per_pos = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)  # [B, L]
        
        # Mask and average over all valid positions across the batch
        mask = attention_mask.float()  # [B, L]
        masked_kl = kl_per_pos * mask
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
            teacher_format = batch.get("teacher_format", "logprobs")  # 'logits' or 'logprobs'
        
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
                    teacher_format=teacher_format,
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
        loss.backward()
        
        return {
            "loss": loss.item(),
            "context_norm": lora_output["context"].norm().item(),
        }
    
    def train_step_parallel(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Batched training step — single student forward pass for the entire batch.
        
        Uses einsum-based batched LoRA hooks so that each sample in the batch
        gets its own LoRA weights applied, while sharing a single student
        forward pass. This is ~Bx faster than sequential per-sample passes.
        """
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
            teacher_format = batch.get("teacher_format", "logprobs")
        else:
            teacher_format = "logprobs"
        
        # Get prompt embeddings
        with torch.no_grad():
            prompt_embeds = self.get_student_embeddings(prompt_ids)
        
        # Hypernetwork generates ALL LoRAs at once (already batched)
        lora_output = self.hypernetwork(prompt_embeds, prompt_mask)
        lora_dict = lora_output["lora_dict"]  # Dict[name -> (A[B,in,r], B[B,r,out])]
        
        # Single batched student forward pass with per-sample LoRA
        with self.lora_injector.apply_batched_lora(lora_dict), autocast("cuda", dtype=torch.bfloat16):
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits  # [B, L, V]
        
        # Compute loss
        if has_cached_teacher:
            loss = self._compute_batched_sparse_kl_loss(
                student_logits,
                teacher_values,
                teacher_indices,
                attention_mask,
                teacher_format=teacher_format,
            )
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            loss = compute_distillation_loss(
                student_logits,
                teacher_outputs.logits,
                attention_mask,
                top_k=self.config.top_k_logits,
                temperature=self.config.temperature,
            )
        
        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
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
            metrics = self.train_step_parallel(batch)
            total_loss += metrics["loss"]
            num_steps += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients (no scaler needed for bfloat16)
                torch.nn.utils.clip_grad_norm_(
                    self.hypernetwork.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Optimizer step
                self.optimizer.step()
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
    
    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """Run validation using batched LoRA forward passes."""
        self.hypernetwork.eval()
        
        total_loss = 0.0
        num_steps = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompt_ids = batch["prompt_ids"].to(self.device)
            prompt_mask = batch["prompt_mask"].to(self.device)
            
            has_cached_teacher = (
                self.config.use_cached_teacher
                and "teacher_values" in batch
                and "teacher_indices" in batch
            )
            
            if has_cached_teacher:
                teacher_values = batch["teacher_values"].to(self.device)
                teacher_indices = batch["teacher_indices"].to(self.device)
                teacher_format = batch.get("teacher_format", "logprobs")
            
            prompt_embeds = self.get_student_embeddings(prompt_ids)
            lora_output = self.hypernetwork(prompt_embeds, prompt_mask)
            lora_dict = lora_output["lora_dict"]
            
            # Batched forward pass
            with self.lora_injector.apply_batched_lora(lora_dict), autocast("cuda", dtype=torch.bfloat16):
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs.logits
            
            if has_cached_teacher:
                loss = self._compute_batched_sparse_kl_loss(
                    student_logits,
                    teacher_values,
                    teacher_indices,
                    attention_mask,
                    teacher_format=teacher_format,
                )
            else:
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = compute_distillation_loss(
                    student_logits,
                    teacher_outputs.logits,
                    attention_mask,
                    top_k=self.config.top_k_logits,
                    temperature=self.config.temperature,
                )
            
            total_loss += loss.item()
            num_steps += 1
        
        self.hypernetwork.train()
        
        avg_val_loss = total_loss / max(num_steps, 1)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        return {"val_loss": avg_val_loss}
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save training checkpoint including scheduler state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "hypernetwork_state_dict": self.hypernetwork.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            "config": self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore scheduler state if available
        if checkpoint.get("scheduler_state_dict") and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored scheduler state")
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
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
        
        # Create full dataloader (shuffle=False for splitting)
        full_dataloader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_prompt_tokens=self.config.max_prompt_tokens,
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            teacher_cache_dir=self.config.teacher_cache_dir if self.config.use_cached_teacher else None,
            shuffle=False,
        )
        
        # Split into train/val
        full_dataset = full_dataloader.dataset
        val_size = max(1, int(len(full_dataset) * self.config.val_split))
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info(f"Dataset split: {train_size} train, {val_size} val")
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        from .data_loader import collate_fn_with_teacher
        collate_fn = lambda batch: collate_fn_with_teacher(
            batch, pad_token_id, self.config.max_trajectory_tokens,
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        # Setup scheduler: warmup → cosine decay
        steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
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
        best_val_loss = float("inf")
        
        for epoch in range(self.config.epochs):
            
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")
            
            metrics = self.train_epoch(train_dataloader, epoch)
            
            logger.info(f"Epoch {epoch + 1} complete. Avg Loss: {metrics['avg_loss']:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.val_every_epochs == 0:
                val_metrics = self.validate(val_dataloader)
                if wandb.run:
                    wandb.log({"val_loss": val_metrics["val_loss"], "epoch": epoch + 1})
                
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_path = Path(self.config.save_dir) / "best.pt"
                    self.save_checkpoint(epoch + 1, str(best_path))
                    logger.info(f"New best val loss: {best_val_loss:.4f}")
            
            # Save periodic checkpoint
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
