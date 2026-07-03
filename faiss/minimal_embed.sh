#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_2011109
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=320G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module load python-vllm
source ../.vllm0.18_venv/bin/activate

srun python embed.py
