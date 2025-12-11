#!/bin/bash
#SBATCH --job-name=syn_find
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

source .venv_pt2.5_merge/bin/activate

merge_res_base="../results/disambiguate_merges/sim_2/results"

srun python3 trace_disambiguation_lineages.py \
  --original "../results/new_descriptors/all_descriptors_new.jsonl" \
  --final "$merge_res_base/all_disambig.jsonl" \
  --lineage "$merge_res_base/combined_lineage.jsonl"\
  --out "../results/disambiguate_merges/traced_lineages/enriched_counts_round_1.jsonl" \
  