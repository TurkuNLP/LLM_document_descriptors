#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err

set -euo pipefail

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

#!/bin/bash
set -Eeuo pipefail

echo "Splitting dataset into shards of $CHUNK_SIZE lines..."

[ -f "$DATASET_PATH" ] || { echo "Error: Dataset file not found: $DATASET_PATH"; exit 1; }

# Create a temporary directory for splitting
TEMP_DIR=$(mktemp -d -p "$SHARD_DIR" "${DATASET_BASENAME}_split_tmp_XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT  # Cleanup on script exit

# Split into temp directory
split -l "$CHUNK_SIZE" -d --additional-suffix=.jsonl \
    "$DATASET_PATH" "$TEMP_DIR/${DATASET_BASENAME}_"

# Rename shards from temp to final directory
i=0
for f in "$TEMP_DIR/${DATASET_BASENAME}_"*.jsonl; do
    [ -e "$f" ] || { echo "Error: No shard files found in temp directory"; exit 1; }
    mv "$f" "$SHARD_DIR/${DATASET_BASENAME}_${i}.jsonl"
    i=$((i+1))
done

# Cleanup temp directory (trap already handles this, but explicit for clarity)
rm -rf "$TEMP_DIR"

echo "Done splitting."