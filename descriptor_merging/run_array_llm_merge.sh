#!/bin/bash
#SBATCH --job-name=llm_merge
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=1-00:00:00
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

source ../.venv_pt2.5/bin/activate

set -euo pipefail

# Print the task index.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "ROCR visible devices: " $ROCR_VISIBLE_DEVICES

BASEDIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${BASEDIR}/../results/new_descriptors/splits"
RUN_ID_BASE="merge_array"

echo "================================"
echo "Reading data from $DATA_DIR"
echo "================================"

# Get the list of .jsonl files into an array
mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.jsonl')
echo "Found ${#FILES[@]} .jsonl files"
for f in "${FILES[@]}"; do echo "  - $f"; done


idx="${SLURM_ARRAY_TASK_ID}"
if (( idx < 0 || idx >= ${#FILES[@]} )); then
  echo "Error: index ${idx} out of range (found ${#FILES[@]} files)" >&2
  exit 1
fi

INPUT_FILE="${FILES[$idx]}"
RUN_ID="${RUN_ID_BASE}_${idx}"

echo "Processing file: $INPUT_FILE"
echo "Run ID: $RUN_ID"

gpu-energy --save || true

srun -n 1 python3 merge_duplicates.py \
  --run-id="${RUN_ID}" \
  --batch-size=512 \
  --data-path="${INPUT_FILE}"

gpu-energy --diff || true
