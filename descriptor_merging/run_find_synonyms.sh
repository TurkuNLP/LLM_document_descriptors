#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000353
#SBATCH --partition=dev-g
#SBATCH --time=00:20:00
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

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

rocm-smi

gpu-energy --save

run_id="synonym_merge_test1"

srun python3 find_synonyms.py --run-id=$run_id \
                              --input="../results/LLM_merges/all_merge_arrays/merge_array_0_merged.jsonl" \
                              --test

gpu-energy --diff
