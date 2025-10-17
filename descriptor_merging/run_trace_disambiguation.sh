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

srun python3 trace_disambiguation.py \
  --id-map ../results/LLM_merges/merge_array_concat/merge_array_concat_merged_ids.jsonl \
  --stage split1 ../results/LLM_merges/merge_array_0/merge_log_iter_*.jsonl \
  --stage split2 ../results/LLM_merges/merge_array_1/merge_log_iter_*.jsonl \
  --stage split3 ../results/LLM_merges/merge_array_2/merge_log_iter_*.jsonl \
  --stage split4 ../results/LLM_merges/merge_array_3/merge_log_iter_*.jsonl \
  --stage split5 ../results/LLM_merges/merge_array_4/merge_log_iter_*.jsonl \
  --stage final ../results/LLM_merges/merge_array_concat/merge_log_iter_*.jsonl \
  --out id_to_original_texts.jsonl \
  --groups-jsonl merged_groups.jsonl \
  --out-clusters cluster_lineage.jsonl
