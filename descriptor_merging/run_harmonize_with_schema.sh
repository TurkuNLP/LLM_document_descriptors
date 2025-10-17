#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

gpu-energy --save

run_id="harmonize_bbc_test"

srun python3 harmonize_with_schema.py --run-id=$run_id \
                                      --input="../results/benchmarks/bbc_news/descriptors_bbc_news.jsonl" \
                                      --schema="../results/schema.jsonl"

gpu-energy --diff
