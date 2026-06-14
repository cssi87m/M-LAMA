#!/bin/bash
#SBATCH --job-name=Speech       # Job name
#SBATCH --output=output.txt      # Output file
#SBATCH --error=error.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodelist=dgx02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
export WANDB_API_KEY="072fb112587c6b4507f5ec59e575d234c3e22649"

conda init
conda activate dinhson

cd /home/user06/Interspeech_2026/model_old
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job CPUs per node: $SLURM_JOB_CPUS_PER_NODE"
# Run script with GPU
taskset -c 100-164 python /home/user06/Interspeech_2026/model_old/train_W2VAudio_bycandidates_V2.py