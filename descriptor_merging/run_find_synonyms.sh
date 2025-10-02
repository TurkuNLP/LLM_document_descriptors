#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
########SBATCH --array=0-4

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

rocm-smi

gpu-energy --save

arr_idx=$SLURM_ARRAY_TASK_ID

run_id="cont_combined_test"

srun python3 find_synonyms.py --run-id=$run_id \
                              --input="../results/synonym_merges/cont_combined/cont_combined.jsonl" \
                              --llm-batch-size 2048 \
                              --min-similarity 0.65 \
                              --max-iters 20 \

gpu-energy --diff
