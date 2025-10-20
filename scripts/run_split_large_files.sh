#!/bin/bash
#SBATCH --job-name=split_files
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# === Load Required Modules ===
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Give input dir for directory of files you want to split.
# Give split-count for how many files you want to split each file in input-dir
# Resulting files will be named <original_file_name>_<split_number>.jsonl

srun python3 ../doc_descriptors/split_large_files.py \
    --input "${SLURM_SUBMIT_DIR}/../results/LLM_merges/merge_array_concat/merge_array_concat_merged_ids.jsonl" \
    --output-dir "${SLURM_SUBMIT_DIR}/../results/synonym_merges/to_be_merged/splits" \
    --split-count 5 \
    --shuffle \
    --seed 42