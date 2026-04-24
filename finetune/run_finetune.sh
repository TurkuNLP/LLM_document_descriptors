#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

source .venv/bin/activate

# Memory management
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

run_id=${1:-"finetune_$SLURM_JOB_ID"}

data_dir="/flash/project_462000963/users/tarkkaot/preprocessed/HPLT4pre-no-eng_8k/"

srun accelerate launch \
    --config_file accelerate/accelerate_config.yaml \
    finetune_qwen.py \
    --run-id=$run_id \
    --data-dir=$data_dir \
    --fast-holdout \
    --use-wandb \
    --group-by-length \
    --add-length-column \
    --filter-by-length 8192 \
    