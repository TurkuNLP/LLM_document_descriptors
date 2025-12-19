#!/bin/bash
#SBATCH --job-name=harmonize
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --array=0-9

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn

benchmark_name="$1"

BASEDIR="${SLURM_SUBMIT_DIR}"
INPUT_DIR="${BASEDIR}/../results/benchmarks/${benchmark_name}/splits"
SCHEMA_FILE="${BASEDIR}/../results/final_schema/descriptor_schema.jsonl"
RUN_ID_BASE="descriptors_${benchmark_name}"

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "================================"
echo "Reading data from $DATA_DIR"
echo "================================"

# Get a sorted list of .jsonl files into an array
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jsonl' | sort -V)
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

srun python3 harmonize_with_schema.py --run-id=$RUN_ID \
                                      --input="${INPUT_FILE}" \
                                      --schema="${SCHEMA_FILE}" \
                                      --cache-db="../results/harmonized/${RUN_ID}/decision_cache.db" \
                                      --topk=10 \
                                      --min-embed-score=0.5 \

gpu-energy --diff || true
