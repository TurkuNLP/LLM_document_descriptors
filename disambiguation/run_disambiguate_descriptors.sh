#!/bin/bash
#SBATCH --job-name=llm_merge
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --array=0-19

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Use flag --resume to continue previous run
# Use flag --test to do a test run on 10k descriptors


# USE THIS FOR ARRAY RUNS! COMMENT OUT OTHERWISE
# Print the task index.
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

BASEDIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${BASEDIR}/../results/new_descriptors/grouped/splits"
RUN_ID_BASE="disambig_1"

echo "================================"
echo "Reading data from $DATA_DIR"
echo "================================"

# Get a sorted list of .jsonl files into an array
mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' | sort -V)
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

srun -n 1 python3 disambiguate_descriptors.py \
  --run-id="${RUN_ID}" \
  --data-path="${INPUT_FILE}" \

gpu-energy --diff || true



# gpu-energy --save

# run_id="disambig_new_test2"

# srun python3 disambiguate_descriptors.py --run-id=$run_id \
#                                          --data-path="../results/new_descriptors/grouped/grouped_descriptors.jsonl" \
#                                         #--resume \
# gpu-energy --diff
