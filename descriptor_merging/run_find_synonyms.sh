#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

rocm-smi

gpu-energy --save

run_id="synonym_big_test2"

srun python3 find_synonyms.py --run-id=$run_id \
                              --input="../results/LLM_merges/all_merge_arrays/merge_array_0_merged.jsonl" \
                              --test 100_000 \
                              --llm-batch-size 1024

gpu-energy --diff
