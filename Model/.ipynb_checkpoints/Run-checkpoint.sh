#!/bin/bash
#SBATCH --job-name=Test+speech       # Job name
#SBATCH --output=output.txt      # Output file
#SBATCH --error=error.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodelist=dgx02
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
export WANDB_API_KEY="072fb112587c6b4507f5ec59e575d234c3e22649"
nvidia-smi
conda init
conda activate dinhson

cd /home/user06/Interspeech_2026/Model
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job CPUs per node: $SLURM_JOB_CPUS_PER_NODE"
# Run script with GPU
python train.py --config config.yaml --checkpoint /home/user01/aiotlab/sondinh/user6/Model/checkpoints_non_hir/model_best_mae_V2_final_score_512dfuse_focal_ranking_nonhir_pretrain_from_olddata.pth
python test.py --config config.yaml --checkpoint /home/user06/Interspeech_2026/Model/Model/checkpoints_finetune_acc1_0/model_best_acc1_0_V2_finetune_acc1_0_balanced_penalties.pth