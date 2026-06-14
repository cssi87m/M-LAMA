#!/usr/bin/env python3
"""
Quick check: Load model and verify parameter setup
"""

import sys
import inspect
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from model import ESLGradingModelByCandidatesWithAudio
from check_model_params import check_model_parameters, compare_with_checkpoint

print("="*80)
print("QUICK MODEL PARAMETER CHECK")
print("="*80)

# Load config
print("\n1. Loading config...")
cfg = Config.from_yaml("config.yaml")
print(f"   ✓ Config loaded")
print(f"   - Text LoRA: {cfg.model.text_encoder_use_lora}")
print(f"   - Audio LoRA: {cfg.model.audio_encoder_use_lora}")
print(f"   - Audio frozen: {cfg.model.audio_encoder_frozen}")

# Build model
print("\n2. Building model...")
sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
valid_params = set(sig.parameters.keys()) - {'self'}
model_kwargs = {k: v for k, v in cfg.model.__dict__.items() if k in valid_params}
model = ESLGradingModelByCandidatesWithAudio(**model_kwargs)
print(f"   ✓ Model built")

# Check parameters
print("\n3. Checking parameters...")
stats = check_model_parameters(model, print_output=True)

# Verify expectations for full fine-tune
print("\n4. Verification for FULL FINE-TUNE:")
checks_passed = True

if stats['has_lora']:
    print("   ❌ FAIL: LoRA parameters detected (should be 0)")
    checks_passed = False
else:
    print("   ✅ PASS: No LoRA parameters")

if stats['text_encoder']['trainable_pct'] < 95:
    print(f"   ❌ FAIL: Text encoder only {stats['text_encoder']['trainable_pct']:.1f}% trainable")
    checks_passed = False
else:
    print(f"   ✅ PASS: Text encoder {stats['text_encoder']['trainable_pct']:.1f}% trainable")

if stats['audio_encoder']['trainable_pct'] < 95:
    print(f"   ❌ FAIL: Audio encoder only {stats['audio_encoder']['trainable_pct']:.1f}% trainable")
    checks_passed = False
else:
    print(f"   ✅ PASS: Audio encoder {stats['audio_encoder']['trainable_pct']:.1f}% trainable")

if stats['trainable_pct'] < 80:
    print(f"   ❌ FAIL: Total only {stats['trainable_pct']:.1f}% trainable (expected >80%)")
    checks_passed = False
else:
    print(f"   ✅ PASS: Total {stats['trainable_pct']:.1f}% trainable")

# Check checkpoint compatibility
if cfg.checkpoint.load_checkpoint:
    ckpt_path = cfg.checkpoint.load_checkpoint
    if Path(ckpt_path).exists():
        print(f"\n5. Checking checkpoint compatibility...")
        compat = compare_with_checkpoint(model, ckpt_path)

        if compat['match_pct'] < 95:
            print(f"   ❌ FAIL: Only {compat['match_pct']:.1f}% match")
            checks_passed = False
        else:
            print(f"   ✅ PASS: {compat['match_pct']:.1f}% match")
    else:
        print(f"\n5. ⚠️  Checkpoint not found: {ckpt_path}")
else:
    print(f"\n5. No checkpoint specified in config")

# Final verdict
print("\n" + "="*80)
if checks_passed:
    print("✅✅✅ ALL CHECKS PASSED - Ready for full fine-tune training!")
else:
    print("❌ SOME CHECKS FAILED - Please fix issues before training!")
print("="*80)

# Save stats to file
import json
stats_file = "model_param_stats.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\n✓ Stats saved to: {stats_file}")
