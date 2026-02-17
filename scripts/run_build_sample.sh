#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

base_dir="/scratch/project_462000963/datasets/hplt/4.0/global-dedup"

srun python3 build_sample.py --input "$base_dir/por_Latn" \
                             --output "$base_dir/samples/por_Latn/por_Latn_1M_sample.jsonl" \
                             -n 1000000 \
                             --initial-keep-prob 0.005