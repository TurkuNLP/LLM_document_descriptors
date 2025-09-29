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
#SBATCH --array=0-4

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source .venv_pt2.5_merge/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

rocm-smi

gpu-energy --save

arr_idx=$SLURM_ARRAY_TASK_ID

run_id="synonyms_split_run1_${arr_idx}"

srun python3 find_synonyms.py --run-id=$run_id \
                              --input="../results/synonym_merges/to_be_merged/splits/merge_array_concat_merged_00${arr_idx}.jsonl" \
                              --llm-batch-size 1024 \
                              --min-similarity 0.65

gpu-energy --diff
