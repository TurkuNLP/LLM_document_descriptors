#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-task=8
#SBATCH --array=0-9
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err
#SBATCH --exclusive

chunk=$1

run_id_base="fw10BT_${chunk}"
run_id="${run_id_base}_00${SLURM_ARRAY_TASK_ID}"
model_name="meta-llama/Llama-3.3-70B-Instruct"

BATCH_SIZE=500
DATA_PATH_BASE="/scratch/project_462000963/datasets/fineweb-10BT-sample"
DATA_PATH="${DATA_PATH_BASE}/split_${chunk}_00${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Running with run_id: $run_id"
echo "Using model: $model_name"
echo "Data path: $DATA_PATH"

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
                                --batch-size=$BATCH_SIZE \
                                --num-batches=-1 \
                                --num-rewrites=0 \
                                --start-index=auto \
                                --data-source=$DATA_PATH \
                                --log-similarity"


PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully (exit code: $PYTHON_EXIT_CODE)."
else
    echo "Python script failed with exit code: $PYTHON_EXIT_CODE."
fi

