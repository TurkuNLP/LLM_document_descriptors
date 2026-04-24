#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=project_462000964
#SBATCH --partition=small
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=120G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

data_dir_base="/flash/project_462000963/users/tarkkaot"

run_id="HPLT4pre-no-eng_4k-8k"

srun python preprocess.py \
    --run-id ${run_id} \
    --input-dir ${data_dir_base}/data \
    --output-dir ${data_dir_base}/preprocessed \
    --cache-dir ${data_dir_base}/cache \
    --max-doc-tokens 7500 \
    --min-input-tokens 4096 \
    --max-input-tokens 8192 \
    --shuffle-files \
    --interleave-buffer-size 8 \
    --seed 42 \
    --quick-truncation \
    --drop-long-docs
