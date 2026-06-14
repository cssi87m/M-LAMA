#!/bin/bash
# ============================================================================
# INFERENCE SCRIPT FOR OLD MODEL (Wav2Vec2 + Qwen2)
# ============================================================================
# This script runs inference on the OLD model architecture using the same
# test data as the new model (from Model/infer.sh)
# ============================================================================

export WANDB_API_KEY="072fb112587c6b4507f5ec59e575d234c3e22649"
nvidia-smi

# Activate conda environment
conda init
conda activate dinhson

cd /home/user06/Interspeech_2026

echo "================================"
echo "OLD MODEL INFERENCE"
echo "================================"
echo "Start time: $(date)"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "================================"

# Define base paths
CONFIG_BASE="model_old/config"
OUTPUT_BASE="model_old/preds"
CHECKPOINT="/home/user06/Interspeech_2026/model_old/model/model_V2_grammar_from_vocab_balanced_20251125_165425.pth"

# ============================================================================
# INFERENCE FOR EACH CRITERION
# ============================================================================

echo ""
echo ">>> [1/5] Running inference for GRAMMAR..."
python model_old/test_old_model.py \
    --config ${CONFIG_BASE}/config_grammar.yaml \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_BASE}/preds_grammar

echo ""
echo ">>> [2/5] Running inference for FLUENCY..."
python model_old/test_old_model.py \
    --config ${CONFIG_BASE}/config_fluency.yaml \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_BASE}/preds_fluency

echo ""
echo ">>> [3/5] Running inference for PRONUNCIATION..."
python model_old/test_old_model.py \
    --config ${CONFIG_BASE}/config_pronunciation.yaml \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_BASE}/preds_pronunciation

echo ""
echo ">>> [4/5] Running inference for VOCABULARY..."
python model_old/test_old_model.py \
    --config ${CONFIG_BASE}/config_vocabulary.yaml \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_BASE}/preds_vocabulary

echo ""
echo ">>> [5/5] Running inference for CONTENT..."
python model_old/test_old_model.py \
    --config ${CONFIG_BASE}/config_content.yaml \
    --checkpoint ${CHECKPOINT} \
    --output_dir ${OUTPUT_BASE}/preds_content

echo ""
echo "================================"
echo "All inference tasks completed at: $(date)"
echo "Results saved in ${OUTPUT_BASE}/preds_* directories"
echo "================================"
echo ""
echo "Summary of results:"
echo "  - Grammar:       ${OUTPUT_BASE}/preds_grammar/test_predictions.csv"
echo "  - Fluency:       ${OUTPUT_BASE}/preds_fluency/test_predictions.csv"
echo "  - Pronunciation: ${OUTPUT_BASE}/preds_pronunciation/test_predictions.csv"
echo "  - Vocabulary:    ${OUTPUT_BASE}/preds_vocabulary/test_predictions.csv"
echo "  - Content:       ${OUTPUT_BASE}/preds_content/test_predictions.csv"
echo "================================"
