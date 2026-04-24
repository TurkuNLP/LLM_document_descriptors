#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=1-00:00:00
#SBATCH --nodes=32
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

set -euo pipefail

# Pick rank-0 node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# Memory management
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8



NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
NUM_GPUS=$((8 * NUM_NODES))

echo "----------------------------------------"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "NUM_NODES=$NUM_NODES"
echo "NUM_GPUS=$NUM_GPUS"
echo "----------------------------------------"

run_id=${1:-"finetune_$SLURM_JOB_ID"}

data_dir="/flash/project_462000963/users/tarkkaot/preprocessed/HPLT4pre-no-eng_4k/"

srun --nodes="$NUM_NODES" --ntasks="$NUM_NODES" bash -c '
NODE_RANK=$SLURM_NODEID
echo "Starting node rank ${NODE_RANK} on $(hostname)"

accelerate launch \
  --config_file accelerate/accelerate_config.yaml \
  --num_machines '"$SLURM_JOB_NUM_NODES"' \
  --num_processes '"$NUM_GPUS"' \
  --machine_rank ${NODE_RANK} \
  --main_process_ip '"$MASTER_ADDR"' \
  --main_process_port '"$MASTER_PORT"' \
  finetune_qwen.py \
    --run-id '"$run_id"' \
    --data-dir '"$data_dir"' \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 8 \
    --group-by-length \
    --use-wandb \
    --fast-holdout \
    --dataloader-num-workers 0 \
'