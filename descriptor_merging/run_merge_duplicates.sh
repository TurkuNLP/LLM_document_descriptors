#!/bin/bash
#SBATCH --job-name=llm_merge
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Use flag --resume to continue previous run
# Use flag --test to do a test run on 10k descriptors

gpu-energy --save

run_id="merge_array_concat"

srun python3 merge_duplicates.py --run-id=$run_id \
                                 --chars-per-batch=2_000_000 \
                                 --data-path="../results/LLM_merges/all_merge_arrays" \
                                 #--resume \
gpu-energy --diff
