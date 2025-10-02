#!/bin/bash
#SBATCH --job-name=watt_hours
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

srun python3 ../doc_descriptors/generate_descriptor_ids.py -i ../results/LLM_merges/merge_array_concat/merge_array_concat_merged.jsonl -o ../results/LLM_merges/merge_array_concat/merge_array_concat_merged_ids.jsonl
