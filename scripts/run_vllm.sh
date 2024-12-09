#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_2011109
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
####SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module load pytorch
source /projappl/project_2011109/otto_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set this to match number of GPUs reserved
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Fixes some bug, maybe?

srun python3 vllm_document_descriptors.py
