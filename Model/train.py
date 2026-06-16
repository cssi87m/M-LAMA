"""
Main training script for ESL Speaking Grading Model (text + audio)
- Loads YAML config
- Builds model/tokenizer/audio processor
- Reuses compatible weights from old checkpoints
- Sets up optimizer + linear warmup scheduler
- Trains with MAE as primary metric and evaluates on test set
"""

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlama.modeling import model_kwargs_from_config, move_optimizer_state_to_device
from mlama.reproducibility import configure_tokenizers, set_seed

try:
    from .config import Config
    from .model import ESLGradingModelByCandidatesWithAudio
    from .trainer import ESLTrainerByCandidatesWithAudio
    from .utils import get_param_groups
    from .audio_encoders import AudioEncoderFactory
except ImportError:
    from config import Config
    from model import ESLGradingModelByCandidatesWithAudio
    from trainer import ESLTrainerByCandidatesWithAudio
    from utils import get_param_groups
    from audio_encoders import AudioEncoderFactory


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train ESL Speaking Grading Model")
    default_config = Path(__file__).parent / "config" / "config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config YAML")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args(argv)


def log_trainable_params(model: torch.nn.Module) -> None:
    """Log detailed parameter breakdown"""
    total = 0
    trainable = 0
    frozen = 0

    lora_params = 0
    text_trainable = 0
    text_frozen = 0
    audio_trainable = 0
    audio_frozen = 0
    base_trainable = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total += num

        # Count LoRA params separately
        if "lora_A" in name or "lora_B" in name or "lora_embedding" in name:
            lora_params += num

        if param.requires_grad:
            trainable += num
            if name.startswith("encoder."):
                text_trainable += num
            elif "audio_encoder" in name:
                audio_trainable += num
            else:
                base_trainable += num
        else:
            frozen += num
            if name.startswith("encoder."):
                text_frozen += num
            elif "audio_encoder" in name:
                audio_frozen += num

    print("\n" + "="*80)
    print("PARAMETER SUMMARY")
    print("="*80)
    print(f"Total parameters:     {total:>15,}")
    print(f"Trainable parameters: {trainable:>15,} ({100*trainable/total:>5.1f}%)")
    print(f"Frozen parameters:    {frozen:>15,} ({100*frozen/total:>5.1f}%)")

    print(f"\nBREAKDOWN BY COMPONENT:")
    print(f"  Text Encoder (Qwen2-1.5B):")
    text_total = text_trainable + text_frozen
    if text_total > 0:
        print(f"    Trainable: {text_trainable:>12,} ({100*text_trainable/text_total:>5.1f}%)")
        print(f"    Frozen:    {text_frozen:>12,} ({100*text_frozen/text_total:>5.1f}%)")

    print(f"  Audio Encoder (Whisper):")
    audio_total = audio_trainable + audio_frozen
    if audio_total > 0:
        print(f"    Trainable: {audio_trainable:>12,} ({100*audio_trainable/audio_total:>5.1f}%)")
        print(f"    Frozen:    {audio_frozen:>12,} ({100*audio_frozen/audio_total:>5.1f}%)")

    print(f"  Fusion/Regression:")
    print(f"    Trainable: {base_trainable:>12,}")

    if lora_params > 0:
        print(f"\n⚠️  LoRA parameters detected: {lora_params:,}")
        print(f"   (Should be 0 for full fine-tune!)")
    else:
        print(f"\n✓ No LoRA parameters (full fine-tune mode)")

    print("="*80)




def _load_state_dict_shape_safe(model: torch.nn.Module, state_dict: dict) -> None:
    """
    Load only parameters whose shapes match; skip incompatible keys.
    """
    current = model.state_dict()
    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
    current.update(matched)
    model.load_state_dict(current)
    if skipped:
        print(f"⚠ Skipped {len(skipped)} keys due to shape mismatch: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def load_checkpoint_with_compatibility(model: torch.nn.Module, checkpoint_path: str | None = None,
                                        checkpoint: dict | None = None, device: str = "cpu",
                                        current_encoder_type: str = "wav2vec2"):
    """
    Load checkpoint and reuse compatible components with enhanced encoder change detection.

    NEW: Detects audio encoder type changes (wav2vec2 ↔ whisper) and skips audio components.
    Compatible components: encoders, projections, attention layers with matching shapes.
    New components (e.g., audio_adapter, part embeddings, larger d_fuse) remain randomly initialized.

    Args:
        model: Current model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        current_encoder_type: Current encoder type ("wav2vec2" or "whisper")
    """
    if checkpoint is None:
        if checkpoint_path is None:
            return model
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found: {checkpoint_path}. Skipping load.")
            return model
        print(f"Loading checkpoint for compatibility: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        if checkpoint_path is not None:
            print(f"Loading checkpoint for compatibility: {checkpoint_path}")

    # Extract state dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    # NEW: Detect encoder type change
    ckpt_config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    # Try to get encoder type from config; if missing, infer from keys
    ckpt_encoder_type = ckpt_config.get('audio_encoder_type')
    if ckpt_encoder_type is None:
        # Heuristic: whisper checkpoints have encoder.conv1/encoder.layers.*, wav2vec2 have feature_extractor.*
        keys = state_dict.keys()
        if any(k.startswith('audio_encoder.encoder.conv1') for k in keys):
            ckpt_encoder_type = 'whisper'
        elif any(k.startswith('audio_encoder.feature_extractor') for k in keys):
            ckpt_encoder_type = 'wav2vec2'
        else:
            ckpt_encoder_type = current_encoder_type
    encoder_changed = (ckpt_encoder_type != current_encoder_type)

    if encoder_changed:
        print(f"⚠ Audio encoder changed: {ckpt_encoder_type} → {current_encoder_type}")
        print("  Audio encoder & adapter weights will be randomly initialized")

    current_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    encoder_skipped = []

    for key, value in state_dict.items():
        # NEW: Skip audio components if encoder changed
        if encoder_changed and ('audio_encoder' in key or 'audio_adapter' in key):
            encoder_skipped.append(key)
            continue

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
    if encoder_changed and encoder_skipped:
        print(f"⚠ Skipped {len(encoder_skipped)} audio encoder/adapter weights (encoder changed)")
    if skipped_keys:
        preview = ", ".join(skipped_keys[:5])
        suffix = "..." if len(skipped_keys) > 5 else ""
        print(f"⚠ Skipped {len(skipped_keys)} incompatible weights: {preview}{suffix}")

    return model


def estimate_total_steps(config: Config) -> tuple[int, int]:
    """
    Estimate total optimizer steps for scheduler using cleaned train data length.

    Returns:
        total_steps: total optimizer steps across all epochs
        num_train: number of training samples after cleaning
    """
    train_df = pd.read_csv(config.data.train_path)
    # train_df = clean_dataframe_bycandidates(
    #     train_df,
    #     remove_low_content=False,
    #     filter_scores=True,
    #     criteria=config.data.criteria,
    # )
    num_train = len(train_df)
    if num_train == 0:
        raise ValueError("Training dataset is empty after cleaning.")

    batches_per_epoch = math.ceil(num_train / config.training.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.training.accumulation_steps)
    total_steps = max(1, steps_per_epoch * config.training.epochs)
    return total_steps, num_train


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = Config.from_yaml(args.config)
    runtime = set_seed()
    device = runtime.device

    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"Seed: {runtime.seed}")
    print(f"Criteria: {config.data.criteria} | Primary metric: {config.checkpoint.monitor_metric}")
    print("=" * 80)

    configure_tokenizers(parallelism=False)

    # Tokenizer & audio processor
    tokenizer = build_tokenizer(config.model.model_name)
    # NEW: Use AudioEncoderFactory to get appropriate processor (Wav2Vec2 or Whisper)
    audio_processor = AudioEncoderFactory.get_processor(
        encoder_type=config.model.audio_encoder_type,
        model_id=config.model.audio_encoder_id
    )
    print(f"✓ Tokenizer and {config.model.audio_encoder_type} audio processor loaded.")
    start_epoch = 0
    ckpt = None
    resume_ckpt = args.checkpoint or config.checkpoint.load_checkpoint

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            print(f"⚠ Checkpoint not found at {ckpt_path}, starting from epoch 0.")

    model = ESLGradingModelByCandidatesWithAudio(
        **model_kwargs_from_config(ESLGradingModelByCandidatesWithAudio, config.model)
    )

    log_trainable_params(model)

    # Load checkpoint (model) and maybe resume optimizer/scheduler
    if ckpt is not None:
        model = load_checkpoint_with_compatibility(
            model,
            checkpoint_path=resume_ckpt,
            checkpoint=ckpt,
            device="cpu",
            current_encoder_type=config.model.audio_encoder_type
        )
        # FINE-TUNE MODE: Always start from epoch 0 (don't resume epoch counter)
        start_epoch = 0
        print(f"✓ Fine-tune mode: Starting from epoch {start_epoch} (checkpoint weights loaded)")


    # Optimizer with differential learning rates
    param_groups = get_param_groups(
        model,
        base_lr=config.training.base_lr,
        encoder_lr=config.training.encoder_lr,
        scale_lr=config.training.scale_lr,
        whisper_lr=getattr(config.training, 'whisper_lr', config.training.encoder_lr)
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

    # If resuming and checkpoint has optimizer/scheduler state
    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, device)
                print("✓ Optimizer state loaded from checkpoint.")
            except Exception as e:
                print(f"⚠ Failed to load optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("✓ Scheduler state loaded from checkpoint.")
            except Exception as e:
                print(f"⚠ Failed to load scheduler state: {e}")

    print(f"Train samples (cleaned): {num_train}")
    print(f"Total optimizer steps: {total_steps} | Warmup: {warmup_steps}")

    # Optional W&B
    if config.logging.wandb_enabled and not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=config.logging.wandb_project,
                name=config.experiment_name,
                config=config.to_dict(),
            )
        except Exception as e:
            print(f"⚠ Failed to initialize wandb: {e}")

    # Ensure optimizer state tensors are on the right device before training
    move_optimizer_state_to_device(optimizer, device)

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

    # Test with best model loaded inside trainer
    test_metrics = trainer.test()
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save final state (best is already saved during training)
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
