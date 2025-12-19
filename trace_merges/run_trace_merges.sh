#!/bin/bash
#SBATCH --job-name=trace_disambig
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

srun python3 trace_merges.py \
  --original "../results/new_descriptors/all_descriptors_new.jsonl" \
  --final "../results/synonym_merges/synonym_merge_2/force_merge_2/force_merge_2.jsonl" \
  --lineage "../results/disambiguate_merges/final_results/final_lineage.jsonl" \
  "../results/synonym_merges/synonym_merge_2/synonym_merge_2_lineage.jsonl" \
  "../results/synonym_merges/synonym_merge_2/force_merge_2/force_merge_2_lineage.jsonl" \
  --out "../results/traced_lineages/final_trace.jsonl" \
  