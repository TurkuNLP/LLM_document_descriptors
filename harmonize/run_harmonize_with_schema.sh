#!/bin/bash
#SBATCH --job-name=harmonize
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err
#SBATCH --array=1
#SBATCH --exclusive

data_name="$1"

BASEDIR="${SLURM_SUBMIT_DIR}"
INPUT_DIR="${BASEDIR}/../results/HPLT4/Qwen3/${data_name}"
SCHEMA_FILE="${BASEDIR}/../results/final_schema/descriptor_schema.jsonl"
RUN_ID_BASE="descriptors_${data_name}"

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "================================"
echo "Reading data from $INPUT_DIR"
echo "================================"

# Get a sorted list of .jsonl files into an array
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 2 -type f -name '*.jsonl' | sort -V)
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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/scratch/project_462000963/users/tarkkaot/containers/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

# This fixes RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
echo HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES

# Set this to avoid errors.
export TORCH_COMPILE_DISABLE=1

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

srun singularity run --rocm --bind /scratch/project_462000963 \
    $SIF bash -c "source ../.aif-venv/bin/activate && python harmonize_with_schema.py \
    --run-id=${RUN_ID}  \
    --input=${INPUT_FILE} \
    --schema=${SCHEMA_FILE} \
    --cache-db=../results/harmonized/${RUN_ID}/decision_cache.db \
    --topk=10 \
    --min-embed-score=0.5 \
    --min-rerank-score=0.5 \
    --embedder=qwen \
    --use-reranker \
    --schema-embed-path=../results/final_schema/schema_embeddings_qwen.npz \
    --reranker=Qwen/Qwen3-Reranker-8B"
