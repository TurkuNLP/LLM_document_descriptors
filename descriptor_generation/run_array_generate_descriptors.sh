#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gpus-per-task=8
#SBATCH --array=0-39
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err
#SBATCH --exclusive

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Memory management
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

run_id_base="ita_Latn"
run_id="${run_id_base}_${SLURM_ARRAY_TASK_ID}"

BATCH_SIZE=500
NUM_BATCHES=50
CHUNK_SIZE=$((BATCH_SIZE * NUM_BATCHES))

DATASET_PATH="../data/HPLT4/global-dedup/ita_Latn_decompressed/batch_1.jsonl"
DATASET_BASENAME="$(basename "$DATASET_PATH" .jsonl)"
SHARD_DIR="../data/HPLT4/global-dedup/ita_Latn_decompressed/dataset_shards"

mkdir -p "$SHARD_DIR"

echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing chunk size: $CHUNK_SIZE"

# -----------------------
# Split dataset (only once)
# -----------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Splitting dataset into shards of $CHUNK_SIZE lines..."

    split -l "$CHUNK_SIZE" -d --additional-suffix=.jsonl \
        "$DATASET_PATH" "$SHARD_DIR/${DATASET_BASENAME}_"

    # Rename to dataset_name_0.jsonl, dataset_name_1.jsonl, ...
    i=0
    for f in "$SHARD_DIR/${DATASET_BASENAME}_"*.jsonl; do
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
while [ ! -f "$SHARD_PATH" ]; do
    sleep 5
done

echo "Using shard: $SHARD_PATH"


srun python3 generate_descriptors.py \
    --run-id="$run_id" \
    --temperature=0.1 \
    --batch-size="$BATCH_SIZE" \
    --num-batches="$NUM_BATCHES" \
    --num-rewrites=0 \
    --start-index=0 \
    --data-source="$SHARD_PATH"
    # --text-column="comment_text"

gpu-energy --diff
