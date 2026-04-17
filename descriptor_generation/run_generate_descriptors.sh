#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

set -eou pipefail

# possible models:
# 'Qwen/Qwen3-Next-80B-A3B-Instruct'
# 'openai/gpt-oss-120b'
# 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'

run_id="qwen3-fw-10BT"
model_name='Qwen/Qwen3-Next-80B-A3B-Instruct'
data_source="fineweb"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/scratch/project_462000963/users/tarkkaot/containers/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

# This fixes RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# Set this to avoid errors.
export TORCH_COMPILE_DISABLE=1

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

srun singularity run --rocm --bind /scratch/project_462000963 \
    $SIF bash -c "source ../.aif-venv/bin/activate && python generate_descriptors.py \
                                --run-id=$run_id \
                                --model=$model_name \
                                --temperature=0.1 \
                                --batch-size=500 \
                                --num-batches=500 \
                                --num-rewrites=0 \
                                --start-index='auto' \
                                --data-source=$data_source \
                                --log-similarity"