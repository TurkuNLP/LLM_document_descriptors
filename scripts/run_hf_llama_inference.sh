#!/bin/bash
#SBATCH --account=project_2011109
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
####SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module load pytorch
source /projappl/project_2011109/otto_venv/bin/activate

export HF_HOME=/scratch/project_2011109/otto/LLM_data_labelling/hf_cache
export TOKENIZERS_PARALLELISM=false

set -xv  # print the command so that we can verify setting arguments correctly from the logs

srun python3 hf_llama_inference.py $*