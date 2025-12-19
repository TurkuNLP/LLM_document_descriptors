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

# Give input dir for directory of files you want to split.
# Give split-count for how many files you want to split each file in input-dir
# Resulting files will be named <original_file_name>_<split_number>.jsonl

benchmark_name="$1"
split_count="$2"
benchmarks_dir="${SLURM_SUBMIT_DIR}/../results/benchmarks"

srun python3 split_large_files.py \
    --input "${benchmarks_dir}/${benchmark_name}/descriptors_${benchmark_name}.jsonl" \
    --output-dir "${benchmarks_dir}/${benchmark_name}/splits" \
    --split-count "${split_count}"