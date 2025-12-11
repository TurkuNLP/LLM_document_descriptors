#!/bin/bash
#SBATCH --job-name=syn_find
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
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

run_id="force_merge_2"
input_file="${SLURM_SUBMIT_DIR}/../results/synonym_merges/synonym_merge_1/synonym_merge_1.jsonl"

srun python3 force_merge.py --run-id=$run_id \
                              --input=$input_file \

gpu-energy --diff
