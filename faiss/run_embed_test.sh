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


# Run full FAISS search & LLM evaluation pipeline including:
# 1. FAISS search for query.
# 2. LLM-based judgement of descriptor matches.
# 3. LLM-based judgement of document matches.
# Run pipeline for multiple queries by contatenating them with "|".

# Set the number of CPU threads based on cpus-per-task
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
#export OMP_PLACES=cores
#export OMP_PROC_BIND=spread


module purge
module load python-vllm
source ../.vllm_venv/bin/activate

srun python embed.py