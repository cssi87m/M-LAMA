#!/bin/bash
echo "========================================================"
echo "  ABLATION: Long Audio Chunk Importance"
echo "========================================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Host    : $(hostname)"
echo "  Start   : $(date)"
echo "========================================================"
nvidia-smi

conda activate dinhson
cd /home/user06/Interspeech_2026/Model

mkdir -p ablation_longaudio_weight/logs
mkdir -p ablation_longaudio_weight/results

CKPT="Model/checkpoints_fluency/model_best_mae_fluency_fusion_only_from_final_ckpt.pth"
CFG="config/config_fluency.yaml"
OUT="ablation_longaudio_weight/results"

echo ""
echo "  Checkpoint : $CKPT"
echo "  Config     : $CFG"
echo "  Split      : test"
echo "  Output dir : $OUT"
echo "--------------------------------------------------------"

python ablation_longaudio_weight/ablation_chunk_importance.py \
    --skip_progressive_keep \
    --checkpoint  "$CKPT"  \
    --config      "$CFG"   \
    --split       test     \
    --output_dir  "$OUT"

echo ""
echo "========================================================"
echo "  DONE at $(date)"
echo "  Results:"
ls -lh "$OUT/"
echo "========================================================"