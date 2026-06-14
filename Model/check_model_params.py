"""
Utility function to check model parameter statistics
Can be imported or used standalone
"""

import torch
from typing import Dict, Optional


def check_model_parameters(model: torch.nn.Module, print_output: bool = True) -> Dict:
    """
    Check model parameter statistics (total, trainable, frozen)

    Args:
        model: PyTorch model
        print_output: Whether to print formatted output (default: True)

    Returns:
        Dictionary with parameter statistics:
        {
            'total': int,
            'trainable': int,
            'frozen': int,
            'trainable_pct': float,
            'frozen_pct': float,
            'text_encoder': {
                'trainable': int,
                'frozen': int,
                'total': int,
                'trainable_pct': float
            },
            'audio_encoder': {
                'trainable': int,
                'frozen': int,
                'total': int,
                'trainable_pct': float
            },
            'fusion': {
                'trainable': int
            },
            'lora_params': int,
            'has_lora': bool
        }
    """
    # Count parameters
    total = 0
    trainable = 0
    frozen = 0

    lora_params = 0
    text_trainable = 0
    text_frozen = 0
    audio_trainable = 0
    audio_frozen = 0
    fusion_trainable = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total += num

        # Count LoRA params
        if any(x in name for x in ['lora_A', 'lora_B', 'lora_embedding']):
            lora_params += num

        if param.requires_grad:
            trainable += num
            if name.startswith('encoder.'):
                text_trainable += num
            elif 'audio_encoder' in name:
                audio_trainable += num
            else:
                fusion_trainable += num
        else:
            frozen += num
            if name.startswith('encoder.'):
                text_frozen += num
            elif 'audio_encoder' in name:
                audio_frozen += num

    # Calculate percentages
    trainable_pct = 100 * trainable / total if total > 0 else 0
    frozen_pct = 100 * frozen / total if total > 0 else 0

    text_total = text_trainable + text_frozen
    text_trainable_pct = 100 * text_trainable / text_total if text_total > 0 else 0

    audio_total = audio_trainable + audio_frozen
    audio_trainable_pct = 100 * audio_trainable / audio_total if audio_total > 0 else 0

    # Build result dict
    result = {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': trainable_pct,
        'frozen_pct': frozen_pct,
        'text_encoder': {
            'trainable': text_trainable,
            'frozen': text_frozen,
            'total': text_total,
            'trainable_pct': text_trainable_pct
        },
        'audio_encoder': {
            'trainable': audio_trainable,
            'frozen': audio_frozen,
            'total': audio_total,
            'trainable_pct': audio_trainable_pct
        },
        'fusion': {
            'trainable': fusion_trainable
        },
        'lora_params': lora_params,
        'has_lora': lora_params > 0
    }

    # Print formatted output
    if print_output:
        print("\n" + "="*80)
        print("MODEL PARAMETER STATISTICS")
        print("="*80)
        print(f"Total parameters:     {total:>15,}")
        print(f"Trainable parameters: {trainable:>15,} ({trainable_pct:>5.1f}%)")
        print(f"Frozen parameters:    {frozen:>15,} ({frozen_pct:>5.1f}%)")

        print(f"\nBREAKDOWN BY COMPONENT:")

        print(f"  Text Encoder (Qwen2):")
        if text_total > 0:
            print(f"    Total:     {text_total:>12,}")
            print(f"    Trainable: {text_trainable:>12,} ({text_trainable_pct:>5.1f}%)")
            print(f"    Frozen:    {text_frozen:>12,} ({100-text_trainable_pct:>5.1f}%)")
        else:
            print(f"    No text encoder found")

        print(f"  Audio Encoder (Whisper):")
        if audio_total > 0:
            print(f"    Total:     {audio_total:>12,}")
            print(f"    Trainable: {audio_trainable:>12,} ({audio_trainable_pct:>5.1f}%)")
            print(f"    Frozen:    {audio_frozen:>12,} ({100-audio_trainable_pct:>5.1f}%)")
        else:
            print(f"    No audio encoder found")

        print(f"  Fusion/Regression:")
        print(f"    Trainable: {fusion_trainable:>12,}")

        # LoRA detection
        if lora_params > 0:
            print(f"\n⚠️  LoRA DETECTED: {lora_params:,} parameters")
            print(f"   (Training mode: LoRA fine-tune)")
        else:
            print(f"\n✓ No LoRA parameters detected")
            print(f"   (Training mode: Full fine-tune)")

        # Training mode assessment
        print(f"\nTRAINING MODE ASSESSMENT:")
        if text_trainable_pct >= 95 and audio_trainable_pct >= 95 and lora_params == 0:
            print(f"  ✅ FULL FINE-TUNE - All encoders trainable, no LoRA")
        elif lora_params > 0:
            print(f"  ⚙️  LoRA FINE-TUNE - Using parameter-efficient training")
        elif text_trainable_pct < 5 and audio_trainable_pct < 5:
            print(f"  ❄️  FROZEN ENCODERS - Only training fusion/regression")
        else:
            print(f"  ⚠️  MIXED MODE - Some components frozen, some trainable")

        print("="*80)

    return result


def compare_with_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """
    Load checkpoint and compare parameter compatibility

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
    """
    print(f"\n{'='*80}")
    print(f"CHECKPOINT COMPATIBILITY CHECK")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ckpt_state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt

    model_state = model.state_dict()

    # Compare
    matched = []
    mismatched = []
    missing = []
    unexpected = []

    matched_params = 0
    total_params = sum(p.numel() for p in model_state.values())

    for key, value in model_state.items():
        if key in ckpt_state:
            if ckpt_state[key].shape == value.shape:
                matched.append(key)
                matched_params += value.numel()
            else:
                mismatched.append((key, ckpt_state[key].shape, value.shape))
        else:
            missing.append(key)

    for key in ckpt_state.keys():
        if key not in model_state:
            unexpected.append(key)

    # Print results
    print(f"\nRESULTS:")
    print(f"  Matched keys:     {len(matched):>6,}")
    print(f"  Mismatched keys:  {len(mismatched):>6,}")
    print(f"  Missing keys:     {len(missing):>6,}")
    print(f"  Unexpected keys:  {len(unexpected):>6,}")

    match_pct = 100 * matched_params / total_params if total_params > 0 else 0
    print(f"\n  Matched params:   {matched_params:>12,} / {total_params:>12,} ({match_pct:>5.1f}%)")

    # Assessment
    if len(mismatched) == 0 and len(missing) == 0 and len(unexpected) == 0:
        print(f"\n✅ PERFECT MATCH - Checkpoint fully compatible")
    elif match_pct >= 95:
        print(f"\n✅ GOOD MATCH - Minor differences, should work fine")
    elif match_pct >= 80:
        print(f"\n⚠️  PARTIAL MATCH - Some incompatibility, check carefully")
    else:
        print(f"\n❌ POOR MATCH - Major incompatibility!")

    # Show samples of issues
    if mismatched:
        print(f"\nSample mismatched keys (shape mismatch):")
        for item in mismatched[:5]:
            print(f"  {item[0]}")
            print(f"    Checkpoint: {item[1]} → Model: {item[2]}")

    if missing:
        print(f"\nSample missing keys (in model but not in checkpoint):")
        for key in missing[:5]:
            print(f"  {key}")

    if unexpected:
        print(f"\nSample unexpected keys (in checkpoint but not in model):")
        for key in unexpected[:5]:
            print(f"  {key}")

    print("="*80)

    return {
        'matched': len(matched),
        'mismatched': len(mismatched),
        'missing': len(missing),
        'unexpected': len(unexpected),
        'match_pct': match_pct
    }


# Example usage
if __name__ == "__main__":
    import inspect
    from pathlib import Path
    from config import Config
    from model import ESLGradingModelByCandidatesWithAudio

    print("="*80)
    print("MODEL PARAMETER CHECKER")
    print("="*80)

    # Load config
    cfg = Config.from_yaml("config.yaml")

    print(f"\nConfig loaded:")
    print(f"  Text encoder LoRA: {cfg.model.text_encoder_use_lora}")
    print(f"  Audio encoder LoRA: {cfg.model.audio_encoder_use_lora}")
    print(f"  Audio encoder frozen: {cfg.model.audio_encoder_frozen}")

    # Build model
    print(f"\nBuilding model...")
    sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    model_kwargs = {k: v for k, v in cfg.model.__dict__.items() if k in valid_params}
    model = ESLGradingModelByCandidatesWithAudio(**model_kwargs)

    # Check parameters
    stats = check_model_parameters(model, print_output=True)

    # Check checkpoint compatibility if specified
    if cfg.checkpoint.load_checkpoint:
        ckpt_path = cfg.checkpoint.load_checkpoint
        if Path(ckpt_path).exists():
            compat = compare_with_checkpoint(model, ckpt_path)
        else:
            print(f"\n⚠️  Checkpoint not found: {ckpt_path}")

    print("\n✓ Check complete!")
