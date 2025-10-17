#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

srun python3 count_final_desc_frequencies.py \
  --docs ../results/new_descriptors/all_descriptors_new.jsonl \
  --lineage ../results/LLM_merges/id_to_original_texts.jsonl \
  --groups ../results/synonym_merges/merged_groups.jsonl \
  --out-final final_id_counts.jsonl \
  --out-members member_id_counts.jsonl \
  --fuzzy

