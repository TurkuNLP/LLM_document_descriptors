#!/bin/bash
#SBATCH --job-name=group_descs
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

srun python3 extract_descriptor_groups.py \
    --input ../results/disambiguate_merges/merges_with_ids/round_1/results/all_merges_disambig.jsonl\
    --input-format processed \
    --out-dir ../results/disambiguate_merges/merges_with_ids/round_1/results \

