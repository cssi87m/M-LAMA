#!/bin/bash
# Training script for Exp2_notext - Audio-focused with fixed text input

# Activate conda environment if needed
# conda activate dinhson

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Run training with config
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    --config config/config.yaml \
    "$@"
