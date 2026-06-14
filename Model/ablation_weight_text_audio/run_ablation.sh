#!/bin/bash

echo "========================================"
echo "  ABLATION: Text vs Audio Weight Study"
echo "========================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Host     : $(hostname)"
echo "  Start    : $(date)"
echo "========================================"
nvidia-smi

conda activate dinhson
cd /home/user06/Interspeech_2026/Model

mkdir -p ablation_weight_text_audio/logs
mkdir -p ablation_weight_text_audio/results

CKPT="/home/user06/Interspeech_2026/Model/Model/checkpoints_fluency/model_best_mae_fluency_fusion_only_from_final_ckpt.pth"
CFG="config/config_fluency.yaml"
OUT="ablation_weight_text_audio/results"

# ----------------------------------------
# STEP 1: Static weight analysis (no data)
# ----------------------------------------
echo ""
echo ">>> [1/2] Static Weight Analysis (print_weights.py)"
echo "    Checkpoint : $CKPT"
echo "    Config     : $CFG"
echo "----------------------------------------"
python ablation_weight_text_audio/print_weights.py \
    --ckpt   "$CKPT" \
    --config "$CFG"  \
    --out_dir "$OUT"

echo ""
echo ">>> Step 1 finished at $(date)"

# ----------------------------------------
# STEP 2: Ablation with real data
# ----------------------------------------
echo ""
echo ">>> [2/2] Ablation Study (ablation_fusion.py)"
echo "    Split      : test"
echo "    Limit      : all samples"
echo "----------------------------------------"
python ablation_weight_text_audio/ablation_fusion.py \
    --ckpt    "$CKPT" \
    --config  "$CFG"  \
    --out_dir "$OUT"  \
    --split   test

echo ""
echo ">>> Step 2 finished at $(date)"

echo ""
echo "========================================"
echo "  DONE at $(date)"
echo "  Results: $OUT/"
ls -lh "$OUT/"
echo "========================================"
