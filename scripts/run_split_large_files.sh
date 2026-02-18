#!/bin/bash
#SBATCH --job-name=split_files
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:15:00
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

base_dir="/scratch/project_462000963/datasets/hplt/4.0/global-dedup"
lang="por_Latn"

srun python3 split_large_files.py \
    --input "${base_dir}/samples/${lang}/1M_sample.jsonl" \
    --output-dir "${base_dir}/samples/${lang}/splits" \
    --split-count 20 \
    --shuffle \
    --seed 42