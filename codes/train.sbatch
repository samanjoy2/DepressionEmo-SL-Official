#!/bin/bash -l
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH -J bert_adv
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source ~/.bashrc
ENV_NAME="${FGDE_CONDA_ENV:-gpu_torch}"
conda activate "$ENV_NAME"

python codes/run_all_bert_experiments_advanced.py --train_batch_size 64 --eval_batch_size 64 --seeds 67 --num_workers 0
