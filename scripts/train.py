#!/usr/bin/env python3
"""
Main training entry point.

The trainer uses model parallelism (device_map="auto") to shard the student
across available GPUs. This is a single-process script — NOT data-parallel.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import DistillationTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config sections with explicit key mapping
    flat_config = {}
    
    # Maps YAML section.key -> TrainingConfig field name
    # Keys not listed here fall through as-is if they match a TrainingConfig field
    key_mapping = {
        # teacher section
        "teacher.model_id": "teacher_model_id",
        # student section
        "student.model_id": "student_model_id",
        # hypernetwork section
        "hypernetwork.num_layers": "num_encoder_layers",
        # checkpointing section
        "checkpointing.save_dir": "save_dir",
        "checkpointing.save_every_epochs": "save_every_epochs",
        # logging section
        "logging.wandb_project": "wandb_project",
        "logging.wandb_run_name": "wandb_run_name",
        "logging.wandb_enabled": "wandb_enabled",
        "logging.log_every_steps": "log_every_steps",
    }
    
    dropped_keys = []
    
    for section, values in config_dict.items():
        if isinstance(values, dict):
            for key, value in values.items():
                qualified = f"{section}.{key}"
                # Check explicit mapping first, then try key as-is
                target_key = key_mapping.get(qualified, key)
                if hasattr(TrainingConfig, target_key):
                    flat_config[target_key] = value
                else:
                    dropped_keys.append(qualified)
    
    if dropped_keys:
        logger.warning(f"YAML keys not mapped to TrainingConfig (ignored): {dropped_keys}")
    
    return TrainingConfig(**flat_config)


def main():
    parser = argparse.ArgumentParser(description="Train Hypernetwork for Knowledge Distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run setup only, don't train",
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HYPERNETWORK DISTILLATION TRAINING")
    logger.info("=" * 60)
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Print config summary
    logger.info("\nConfiguration:")
    logger.info(f"  Teacher: {config.teacher_model_id}")
    logger.info(f"  Student: {config.student_model_id}")
    logger.info(f"  Hidden Dim: {config.hidden_dim}")
    logger.info(f"  LoRA Rank: {config.lora_rank}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    
    # Create trainer
    trainer = DistillationTrainer(config)
    
    # Setup
    logger.info("\nSetting up trainer...")
    trainer.setup()
    
    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {epoch}")
    
    # Dry run exits here
    if args.dry_run:
        logger.info("\n[DRY RUN] Setup complete. Exiting.")
        return 0
    
    # Train
    logger.info("\nStarting training...")
    trainer.train()
    
    logger.info("\nTraining complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
