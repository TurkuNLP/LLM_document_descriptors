#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-task=8
#SBATCH --array=0-19
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err
#SBATCH --exclusive

lang=$1
run_id_base="${lang}_sample"
run_id="${run_id_base}_${SLURM_ARRAY_TASK_ID}"
model_name='Qwen/Qwen3-Next-80B-A3B-Instruct'

BATCH_SIZE=500
NUM_BATCHES=100
CHUNK_SIZE=$((BATCH_SIZE * NUM_BATCHES))

DATASET_PATH_BASE="/scratch/project_462000963/datasets/hplt/4.0/global-dedup/samples/${lang}"
DATASET_PATH="$DATASET_PATH_BASE/1M_sample.jsonl"
DATASET_BASENAME="$(basename "$DATASET_PATH" .jsonl)"
SHARD_DIR="$DATASET_PATH_BASE/shards"

mkdir -p "$SHARD_DIR"

echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing chunk size: $CHUNK_SIZE"

do_split=1

echo "do_split set to $do_split"
# -----------------------
# Split dataset (only once)
# -----------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ $do_split -eq 1 ]; then
    echo "Splitting dataset into shards of $CHUNK_SIZE lines..."

    [ -f "$DATASET_PATH" ] || { echo "Error: Dataset file not found: $DATASET_PATH"; exit 1; }
    
    split -l "$CHUNK_SIZE" -d --additional-suffix=.jsonl \
        "$DATASET_PATH" "$SHARD_DIR/${DATASET_BASENAME}_"

    # Rename to dataset_name_0.jsonl, dataset_name_1.jsonl, ...
    i=0
    for f in "$SHARD_DIR/${DATASET_BASENAME}_"*.jsonl; do
        [ -e "$f" ] || { echo "Error: No shard files found"; exit 1; }
        mv "$f" "$SHARD_DIR/${DATASET_BASENAME}_${i}.jsonl"
        i=$((i+1))
    done

    echo "Done splitting."
fi

# -----------------------
# Wait until shards exist
# -----------------------
SHARD_PATH="$SHARD_DIR/${DATASET_BASENAME}_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Waiting for shard: $SHARD_PATH"
timeout 120 bash -c 'while [ ! -f "$1" ]; do sleep 5; done' _ "$SHARD_PATH" || {
    echo "Error: Shard file not found after 120 seconds"
    exit 1
}

# Wait a bit to avoid all jobs requesting model from HF at once.
wait_time=$((SLURM_ARRAY_TASK_ID * 30))
echo "Sleeping for $wait_time seconds to stagger model loading..."
sleep $wait_time

echo "Using shard: $SHARD_PATH"

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
                                --num-batches=$NUM_BATCHES \
                                --num-rewrites=0 \
                                --start-index=auto \
                                --data-source=$SHARD_PATH \
                                --log-similarity"

PYTHON_EXIT_CODE=$?

# Check the exit code
# If the Python script succeeded, we can safely remove the shard file. If it failed, we keep the shard for reruns.
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully (exit code: $PYTHON_EXIT_CODE). Removing shard file..."
    rm -f "$SHARD_PATH"
    echo "Shard file removed."

    # On success, move results to final location
    echo Moving results to final location...
    mkdir -p ../results/HPLT4/Qwen3/${run_id_base}
    mv "../results/${run_id}" ../results/HPLT4/Qwen3/${run_id_base}/
else
    echo "Python script failed with exit code: $PYTHON_EXIT_CODE. Keeping shard file."
    exit $PYTHON_EXIT_CODE
fi

