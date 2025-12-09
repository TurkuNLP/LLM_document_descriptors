#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
######SBATCH --array=0-4

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

gpu-energy --save

arr_idx=$SLURM_ARRAY_TASK_ID

run_id="synonym_merge_1"
input_file="${SLURM_SUBMIT_DIR}/../results/disambiguate_merges/merges_with_ids/round_5/results/all_merges_disambig.jsonl"

srun python3 find_synonyms.py --run-id=$run_id \
                              --input=$input_file \
                              --llm-batch-size 1024 \
                              --max-iters 30 \

gpu-energy --diff
