#!/bin/bash
export WANDB_API_KEY="072fb112587c6b4507f5ec59e575d234c3e22649"
nvidia-smi
conda init


# conda activate audioflamingo2
# cd /home/user06/Baseline/Flamingo_3

# python evaluate.py   --model_path /home/user06/Baseline/Flamingo_3/Fla3/best  --val_jsonl /home/user06/Baseline/Flamingo_3/data/val_multi.jsonl   --criterion final   --output_csv /home/user06/Baseline/Flamingo_3/Fla3/results_test.csv

# echo ""
# echo ">>> Done evaluating Flamingo-3 model on final criterion."

conda activate dinhson

cd /home/user06/Interspeech_2026/Model
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job CPUs per node: $SLURM_JOB_CPUS_PER_NODE"

# Define base paths
CHECKPOINT_BASE="/home/user06/Interspeech_2026/Model/Model"
OUTPUT_BASE="Model/Preds"

# ============================================================================
# INFERENCE FOR EACH CRITERION
# ============================================================================

echo ""
echo ">>> [1/6] Running inference for GRAMMAR..."
python test_new.py \
    --output_dir ${OUTPUT_BASE}_grammar \
    --config config/config_grammar.yaml \
    --checkpoint ${CHECKPOINT_BASE}/checkpoints_grammar/model_best_acc1_0_grammar_fusion_only_from_final_ckpt.pth

echo ""
echo ">>> [2/6] Running inference for FLUENCY..."
python test_new.py \
    --output_dir ${OUTPUT_BASE}_fluency \
    --config config/config_fluency.yaml \
    --checkpoint ${CHECKPOINT_BASE}/checkpoints_fluency/model_best_acc1_0_fluency_fusion_only_from_final_ckpt.pth

echo ""
echo ">>> [3/6] Running inference for PRONUNCIATION..."
python test_new.py \
    --output_dir ${OUTPUT_BASE}_pronunciation \
    --config config/config_pronunciation.yaml \
    --checkpoint ${CHECKPOINT_BASE}/checkpoints_pronunciation/model_best_acc1_0_pronunciation_fusion_only_from_final_ckpt.pth

echo ""
echo ">>> [4/6] Running inference for VOCABULARY..."
python test_new.py \
    --output_dir ${OUTPUT_BASE}_vocabulary \
    --config config/config_vocabulary.yaml \
    --checkpoint ${CHECKPOINT_BASE}/checkpoints_vocabulary/model_best_acc1_0_vocabulary_fusion_only_from_final_ckpt.pth

echo ""
echo ">>> [5/6] Running inference for CONTENT..."
python test_new.py \
    --output_dir ${OUTPUT_BASE}_content \
    --config config/config_content.yaml \
    --checkpoint ${CHECKPOINT_BASE}/checkpoints_content/model_best_acc1_0_content_fusion_only_from_final_ckpt.pth

echo ""
echo "================================"
echo "All inference tasks completed at: $(date)"
echo "Results saved in ${OUTPUT_BASE}_* directories"
echo "================================"

# Optional: Combine results
# python combine_predictions.py --input_dir ${OUTPUT_BASE}_* --output combined_predictions.csv