#!/bin/bash
#SBATCH --job-name=vllm_test
#SBATCH --account=project_2011109
#SBATCH --partition=gpusmall
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:2
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module load pytorch
source /projappl/project_2011109/otto_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1  # Set this to match number of GPUs reserved

srun python3 vllm_document_keywords.py
