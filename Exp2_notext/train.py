"""
Training script for Exp2_notext: ESL Speaking Grading Model (Audio-focused, minimal text)
- Uses fixed placeholder text: "This is work of candidate: "
- Focuses on audio encoder learning
- Loads YAML config and trains with MAE as primary metric
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb

from config import Config
from model import ESLGradingModelByCandidatesWithAudio
from trainer import ESLTrainerByCandidatesWithAudio
from utils import get_param_groups
from audio_encoders import AudioEncoderFactory


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESL Speaking Grading Model (No Text)")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to config YAML")
    parser.add_argument("--no_wandb", action="store_true", 
                       help="Disable Weights & Biases logging")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint to resume from")
    parser.add_argument("--criteria", type=str, default=None,
                       help="Override criteria (e.g., grammar, vocabulary, etc.)")
    return parser.parse_args()


def log_trainable_params(model: torch.nn.Module) -> None:
    """Log detailed parameter breakdown"""
    total = 0
    trainable = 0
    frozen = 0

    text_trainable = 0
    text_frozen = 0
    audio_trainable = 0
    audio_frozen = 0
    base_trainable = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total += num

        if param.requires_grad:
            trainable += num
            if name.startswith("encoder."):
                text_trainable += num
            elif "audio_encoder" in name or "audio_adapter" in name:
                audio_trainable += num
            else:
                base_trainable += num
        else:
            frozen += num
            if name.startswith("encoder."):
                text_frozen += num
            elif "audio_encoder" in name or "audio_adapter" in name:
                audio_frozen += num

    print("\n" + "="*80)
    print("PARAMETER SUMMARY (NOTEXT EXPERIMENT - Fixed Text Input)")
    print("="*80)
    print(f"Total parameters:     {total:>15,}")
    print(f"Trainable parameters: {trainable:>15,} ({100*trainable/total:>5.1f}%)")
    print(f"Frozen parameters:    {frozen:>15,} ({100*frozen/total:>5.1f}%)")

    print(f"\nBREAKDOWN BY COMPONENT:")
    print(f"  Text Encoder (Qwen2-1.5B) - Processing fixed placeholder:")
    text_total = text_trainable + text_frozen
    if text_total > 0:
        print(f"    Trainable: {text_trainable:>12,} ({100*text_trainable/text_total:>5.1f}%)")
        print(f"    Frozen:    {text_frozen:>12,} ({100*text_frozen/text_total:>5.1f}%)")

    print(f"  Audio Encoder + Adapter:")
    audio_total = audio_trainable + audio_frozen
    if audio_total > 0:
        print(f"    Trainable: {audio_trainable:>12,} ({100*audio_trainable/audio_total:>5.1f}%)")
        print(f"    Frozen:    {audio_frozen:>12,} ({100*audio_frozen/audio_total:>5.1f}%)")

    print(f"  Fusion/Regression:")
    print(f"    Trainable: {base_trainable:>12,}")

    print("="*80)


def get_param_groups(model: torch.nn.Module, base_lr: float, encoder_lr: float, 
                     audio_lr: float, scale_lr: float):
    """
    Create parameter groups with different learning rates
    
    Groups:
    1. encoder: Text encoder (Qwen2) - processing fixed text
    2. audio_encoder: Audio encoder
    3. audio_adapter: Audio adapter layers
    4. base: Regression head and other components
    """
    encoder_params = []
    audio_encoder_params = []
    audio_adapter_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_params.append(param)
        elif "audio_encoder." in name and "audio_adapter" not in name:
            audio_encoder_params.append(param)
        elif "audio_adapter" in name:
            audio_adapter_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
            "name": "text_encoder"
        })
    
    if audio_encoder_params:
        param_groups.append({
            "params": audio_encoder_params,
            "lr": audio_lr,
            "name": "audio_encoder"
        })
    
    if audio_adapter_params:
        param_groups.append({
            "params": audio_adapter_params,
            "lr": scale_lr,
            "name": "audio_adapter"
        })
    
    if base_params:
        param_groups.append({
            "params": base_params,
            "lr": base_lr,
            "name": "regression"
        })

    return param_groups


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move all optimizer state tensors to the target device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint_safe(model: torch.nn.Module, checkpoint_path: str, device: str = "cpu"):
    """
    Load checkpoint with shape compatibility check
    
    Args:
        model: Current model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}. Skipping load.")
        return model, None

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    current_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key in current_state:
            if current_state[key].shape == value.shape:
                current_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: {value.shape} → {current_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in new model")

    # Load updated state dict
    model.load_state_dict(current_state)

    print(f"✓ Loaded {len(loaded_keys)} compatible weights")
    if skipped_keys:
        preview = ", ".join(skipped_keys[:5])
        suffix = "..." if len(skipped_keys) > 5 else ""
        print(f"⚠ Skipped {len(skipped_keys)} incompatible weights: {preview}{suffix}")

    return model, checkpoint


def estimate_total_steps(config: Config) -> tuple[int, int]:
    """
    Estimate total optimizer steps for scheduler
    
    Returns:
        total_steps: total optimizer steps across all epochs
        num_train: number of training samples
    """
    train_df = pd.read_csv(config.data.train_path)
    num_train = len(train_df)
    if num_train == 0:
        raise ValueError("Training dataset is empty.")

    batches_per_epoch = math.ceil(num_train / config.training.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.training.accumulation_steps)
    total_steps = max(1, steps_per_epoch * config.training.epochs)
    return total_steps, num_train


def build_tokenizer(model_name: str):
    """Build tokenizer from model name"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main():
    args = parse_args()
    
    # Load config using the Config class
    config = Config.from_yaml(args.config)
    
    # Override criteria if specified
    if args.criteria:
        config.data.criteria = args.criteria
        config.experiment_name = f"exp2_notext_{args.criteria}"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Experiment: {config.experiment_name} (NOTEXT - Fixed Text Input)")
    print(f"Device: {device}")
    print(f"Criteria: {config.data.criteria} | Primary metric: {config.checkpoint.monitor_metric}")
    print(f"Audio encoder: {config.model.audio_encoder_type} - {config.model.audio_encoder_id}")
    print(f"Text encoder: {config.model.model_name} (processing fixed placeholder)")
    print("=" * 80)

    # Set seeds
    set_seed()

    # Tokenizer and audio processor
    tokenizer = build_tokenizer(config.model.model_name)
    audio_processor = AudioEncoderFactory.get_processor(
        encoder_type=config.model.audio_encoder_type,
        model_id=config.model.audio_encoder_id
    )
    print(f"✓ Tokenizer and {config.model.audio_encoder_type} audio processor loaded.")

    # Model - Build with ALL config parameters
    import inspect
    sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    model_kwargs = {k: v for k, v in config.model.__dict__.items() if k in valid_params}
    model = ESLGradingModelByCandidatesWithAudio(**model_kwargs)

    log_trainable_params(model)

    # Load checkpoint if specified
    start_epoch = 0
    ckpt = None
    resume_ckpt = args.checkpoint or config.checkpoint.load_checkpoint
    
    if resume_ckpt:
        model, ckpt = load_checkpoint_safe(model, resume_ckpt, device="cpu")
        if ckpt is not None:
            start_epoch = 0  # Always start from epoch 0 for fine-tuning
            print(f"✓ Checkpoint loaded. Starting from epoch {start_epoch}")

    # Optimizer with differential learning rates
    param_groups = get_param_groups(
        model,
        base_lr=config.training.base_lr,
        encoder_lr=config.training.encoder_lr,
        audio_lr=getattr(config.training, 'whisper_lr', config.training.encoder_lr),
        scale_lr=config.training.scale_lr,
    )
    
    for group in param_groups:
        name = group.get("name", "group")
        lr = group.get("lr", None)
        count = sum(p.numel() for p in group.get("params", []))
        print(f"Param group {name}: lr={lr} params={count:,}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.training.weight_decay)

    # Scheduler
    total_steps, num_train = estimate_total_steps(config)
    warmup_steps = min(config.training.warmup_steps, total_steps)
    if warmup_steps < config.training.warmup_steps:
        print(f"Warmup steps truncated to {warmup_steps} (total steps: {total_steps})")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Load optimizer/scheduler state if resuming
    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _move_optimizer_state_to_device(optimizer, device)
                print("✓ Optimizer state loaded from checkpoint.")
            except Exception as e:
                print(f"⚠ Failed to load optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("✓ Scheduler state loaded from checkpoint.")
            except Exception as e:
                print(f"⚠ Failed to load scheduler state: {e}")

    print(f"Train samples: {num_train}")
    print(f"Total optimizer steps: {total_steps} | Warmup: {warmup_steps}")

    # Initialize W&B if enabled
    if config.logging.wandb_enabled and not args.no_wandb:
        try:
            import wandb
            wandb.login(key="b6bf189f51b29501771e7a3294635dfee6d75021", relogin=True)
            print("✓ Weights & Biases logging initialized.")
            wandb.init(
                project=config.logging.wandb_project,
                name=config.experiment_name,
                config=config.to_dict(),
            )   
            

        except Exception as e:
            print(f"⚠ Failed to initialize wandb: {e}")

    # Ensure optimizer state is on correct device
    _move_optimizer_state_to_device(optimizer, device)

    # Trainer
    trainer = ESLTrainerByCandidatesWithAudio(
        model=model,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # Train
    trainer.train(start_epoch=start_epoch)

    # Test with best model
    test_metrics = trainer.test()
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save final state
    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / f"model_final_{config.experiment_name}.pth"
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config.to_dict(),
        "monitor": config.checkpoint.monitor_metric,
    }, final_path)
    print(f"\nFinal model state saved to: {final_path}")


if __name__ == "__main__":
    main()
