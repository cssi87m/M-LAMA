echo "========================================================"
echo "  ABLATION: Test 5 criterias (only text)"
echo "========================================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Host    : $(hostname)"
echo "  Start   : $(date)"
echo "========================================================"
nvidia-smi

conda activate dinhson
cd /home/user06/Interspeech_2026/Exp2_noaudio

python test.py --config config/config_fluency.yaml
python test.py --config config/config_content.yaml
