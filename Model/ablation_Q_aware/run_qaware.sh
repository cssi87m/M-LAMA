#!/bin/bash


echo "========================================================"
echo "  ABLATION: Question-Aware Cross-Attention (All Criteria)"
echo "========================================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Host    : $(hostname)"
echo "  Start   : $(date)"
echo "========================================================"
nvidia-smi

conda activate dinhson
cd /home/user06/Interspeech_2026/Model

mkdir -p ablation_Q_aware/logs
mkdir -p ablation_Q_aware/results

CKPT_BASE="Model/checkpoints"
CFG_BASE="config"
OUT_BASE="ablation_Q_aware/results"

# ============================================================================
# [1/5] GRAMMAR
# ============================================================================
echo ""
echo ">>> [1/5] GRAMMAR ..."
python ablation_Q_aware/ablation_qaware.py \
    --ckpt    "${CKPT_BASE}_grammar/model_best_acc1_0_grammar_fusion_only_from_final_ckpt.pth" \
    --config  "${CFG_BASE}/config_grammar.yaml" \
    --split   test \
    --out_dir "${OUT_BASE}/grammar" \
    --ablation_only

# ============================================================================
# [2/5] FLUENCY
# ============================================================================
echo ""
echo ">>> [2/5] FLUENCY ..."
python ablation_Q_aware/ablation_qaware.py \
    --ckpt    "${CKPT_BASE}_fluency/model_best_acc1_0_fluency_fusion_only_from_final_ckpt.pth" \
    --config  "${CFG_BASE}/config_fluency.yaml" \
    --split   test \
    --out_dir "${OUT_BASE}/fluency" \
    --ablation_only

# ============================================================================
# [3/5] PRONUNCIATION
# ============================================================================
echo ""
echo ">>> [3/5] PRONUNCIATION ..."
python ablation_Q_aware/ablation_qaware.py \
    --ckpt    "${CKPT_BASE}_pronunciation/model_best_acc1_0_pronunciation_fusion_only_from_final_ckpt.pth" \
    --config  "${CFG_BASE}/config_pronunciation.yaml" \
    --split   test \
    --out_dir "${OUT_BASE}/pronunciation" \
    --ablation_only

# ============================================================================
# [4/5] VOCABULARY
# ============================================================================
echo ""
echo ">>> [4/5] VOCABULARY ..."
python ablation_Q_aware/ablation_qaware.py \
    --ckpt    "${CKPT_BASE}_vocabulary/model_best_acc1_0_vocabulary_fusion_only_from_final_ckpt.pth" \
    --config  "${CFG_BASE}/config_vocabulary.yaml" \
    --split   test \
    --out_dir "${OUT_BASE}/vocabulary" \
    --ablation_only

# ============================================================================
# [5/5] CONTENT
# ============================================================================
echo ""
echo ">>> [5/5] CONTENT ..."
python ablation_Q_aware/ablation_qaware.py \
    --ckpt    "${CKPT_BASE}_content/model_best_acc1_0_content_fusion_only_from_final_ckpt.pth" \
    --config  "${CFG_BASE}/config_content.yaml" \
    --split   test \
    --out_dir "${OUT_BASE}/content" \
    --ablation_only

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================"
echo "  ALL DONE at $(date)"
echo "  Results:"
for CRIT in grammar fluency pronunciation vocabulary content; do
    echo "  --- ${CRIT} ---"
    cat "${OUT_BASE}/${CRIT}/ablated_results.json" 2>/dev/null \
        | python -c "import sys,json; d=json.load(sys.stdin); m=d['metrics_ablated']; \
          print(f'    MAE={m[\"mae\"]:.4f}  QWK={m[\"qwk\"]:.4f}  Acc@1={m[\"acc_1.0\"]:.4f}')" \
        || echo "    (not found)"
done
echo "========================================================"
