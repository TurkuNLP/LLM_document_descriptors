#!/bin/bash
#SBATCH --job-name=llm_merge
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
###SBATCH --array=0-1

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Use flag --resume to continue previous run
# Use flag --test to do a test run on 10k descriptors
# Use flag --mock-llm to use mock LLM responses for quick testing


# USE THIS FOR ARRAY RUNS! COMMENT OUT OTHERWISE
# Print the task index.


BASEDIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${BASEDIR}/../results/disambiguate_merges/round_19/results/grouped"
RUN_ID_BASE="disambig"

RUN_ID="${RUN_ID_BASE}_1"
INPUT_FILE="${DATA_DIR}/grouped_descriptors_with_ids.jsonl"

# echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
# echo "================================"
# echo "Reading data from $DATA_DIR"
# echo "================================"

# # Get a sorted list of .jsonl files into an array
# mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' | sort -V)
# echo "Found ${#FILES[@]} .jsonl files"
# for f in "${FILES[@]}"; do echo "  - $f"; done


# idx="${SLURM_ARRAY_TASK_ID}"
# if (( idx < 0 || idx >= ${#FILES[@]} )); then
#   echo "Error: index ${idx} out of range (found ${#FILES[@]} files)" >&2
#   exit 1
# fi

# INPUT_FILE="${FILES[$idx]}"
# RUN_ID="${RUN_ID_BASE}_${idx}"

# echo "Processing file: $INPUT_FILE"
# echo "Run ID: $RUN_ID"

gpu-energy --save || true

srun -n 1 python3 disambiguate_descriptors.py \
  --run-id="${RUN_ID}" \
  --data-path="${INPUT_FILE}" \
  --cohort-size=0 \

gpu-energy --diff || true
