#!/bin/bash
export WANDB_API_KEY="072fb112587c6b4507f5ec59e575d234c3e22649"
nvidia-smi
conda init
conda activate dinhson

cd /home/user06/Interspeech_2026/Model
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Job CPUs per node: $SLURM_JOB_CPUS_PER_NODE"
# Run script with GPU
# fluency,pronunciation,grammar,vocabulary,content
python train.py --config config/config_grammar.yaml #grammar
python train.py --config config/config_fluency.yaml #fluency
python train.py --config config/config_pronunciation.yaml #pronunciation
python train.py --config config/config_vocabulary.yaml #vocabulary
python train.py --config config/config_content.yaml #content

