#!/bin/bash
#SBATCH --job-name=Speech_test       # Job name
#SBATCH --output=output_test.txt      # Output file
#SBATCH --error=error_test.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1
#SBATCH --nodelist=dgx02
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
export WANDB_API_KEY="b6bf189f51b29501771e7a3294635dfee6d75021"

source ~/.bashrc
# conda init
conda activate dinhson

cd /home/user06/Interspeech_2026/model_old
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job CPUs per node: $SLURM_JOB_CPUS_PER_NODE"
# Run script with GPU
python /home/user06/Interspeech_2026/model_old/train_W2VAudio_bycandidates.py