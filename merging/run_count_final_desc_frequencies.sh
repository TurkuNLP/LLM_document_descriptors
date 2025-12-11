#!/bin/bash
#SBATCH --job-name=count_freqs
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

# srun python3 count_final_desc_frequencies.py \
#   --docs ../results/new_descriptors/all_descriptors_new.jsonl \
#   --lineage ../results/disambiguate_merges/archived_run/id_to_original_texts.jsonl \
#   --groups ../results/synonym_merges/merged_groups.jsonl \
#   --out-final final_id_counts.jsonl \
#   --fuzzy

python count_desc_freqs_new.py \
  --source ../results/new_descriptors/all_descriptors_new.jsonl \
  --lineage-roots ../results/disambiguate_merges/merges_with_ids \
  --finals-roots ../results/disambiguate_merges/merges_with_ids \
  --out-dir ../results/disambiguate_merges/final_desc_frequencies
