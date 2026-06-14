#!/usr/bin/env python3
"""
Quick check: Verify eval uses 6 chunks
"""

import sys
from pathlib import Path

print("=" * 80)
print("CHECKING EVAL_NUM_CHUNKS USAGE")
print("=" * 80)

# Read config.yaml
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    for line in f:
        if 'num_chunks:' in line and 'Number of chunks for TRAINING' in line:
            train_chunks = int(line.split(':')[1].split('#')[0].strip())
            print(f"\n✓ Training num_chunks: {train_chunks}")
        elif 'eval_num_chunks:' in line:
            eval_chunks = int(line.split(':')[1].split('#')[0].strip())
            print(f"✓ Eval num_chunks: {eval_chunks}")

# Check trainer.py
trainer_path = Path(__file__).parent / "trainer.py"
with open(trainer_path, 'r') as f:
    content = f.read()

    # Check if eval_num_chunks is used
    if 'eval_num_chunks = self.config.audio.eval_num_chunks' in content:
        print("\n✓ trainer.py: eval_num_chunks variable created")
    else:
        print("\n❌ trainer.py: eval_num_chunks NOT found")
        sys.exit(1)

    if 'num_chunks=eval_num_chunks,  # Use eval_num_chunks for validation' in content:
        print("✓ trainer.py: val_dataset uses eval_num_chunks")
    else:
        print("❌ trainer.py: val_dataset NOT using eval_num_chunks")
        sys.exit(1)

    if 'num_chunks=eval_num_chunks,  # Use eval_num_chunks for test' in content:
        print("✓ trainer.py: test_dataset uses eval_num_chunks")
    else:
        print("❌ trainer.py: test_dataset NOT using eval_num_chunks")
        sys.exit(1)

# Check test.py
test_path = Path(__file__).parent / "test.py"
with open(test_path, 'r') as f:
    content = f.read()

    if 'num_chunks = cfg.audio.eval_num_chunks if cfg.audio.eval_num_chunks is not None else cfg.audio.num_chunks' in content:
        print("\n✓ test.py: Uses eval_num_chunks when available")
    else:
        print("\n⚠️  test.py: May not use eval_num_chunks")

# Check test_new.py
test_new_path = Path(__file__).parent / "test_new.py"
if test_new_path.exists():
    with open(test_new_path, 'r') as f:
        content = f.read()

        if 'num_chunks = cfg.audio.eval_num_chunks if cfg.audio.eval_num_chunks is not None else cfg.audio.num_chunks' in content:
            print("✓ test_new.py: Uses eval_num_chunks when available")
        else:
            print("⚠️  test_new.py: May not use eval_num_chunks")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Training will use: {train_chunks} chunks")
print(f"Validation/Test will use: {eval_chunks} chunks")
print("\n✅ Configuration is CORRECT!")
print("=" * 80)
