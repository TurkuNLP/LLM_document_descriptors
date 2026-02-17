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

srun python3 build_sample.py --input "../data/HPLT4/global-dedup/ita_Latn" \
                             --output "../data/HPLT4/global-dedup/ita_Latn_sample.jsonl" \
                             -n 1000000 \
                             --initial-keep-prob 0.005