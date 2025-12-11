#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:30:00
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

python trace_synonym_merging.py \
  --run split1 ../results/synonym_merges/synonym_split_run_0 \
  --run split2 ../results/synonym_merges/synonym_split_run_1 \
  --run split3 ../results/synonym_merges/synonym_split_run_2 \
  --run split4 ../results/synonym_merges/synonym_split_run_3 \
  --run split5 ../results/synonym_merges/synonym_split_run_4 \
  --run final1  ../results/synonym_merges/synonym_concat \
  --run final2  ../results/synonym_merges/synonym_concat_cont \
  --all-ids ../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_ids_000.jsonl \
  --all-ids ../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_ids_001.jsonl \
  --all-ids ../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_ids_002.jsonl \
  --all-ids ../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_ids_003.jsonl \
  --all-ids ../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_ids_004.jsonl \
  --out-groups merged_groups.jsonl \
  --out-edges merge_edges.jsonl

