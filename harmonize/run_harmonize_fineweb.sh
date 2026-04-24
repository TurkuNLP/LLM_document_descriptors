#!/bin/bash
#SBATCH --job-name=harmonize
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err
#SBATCH --array=0-9
#SBATCH --exclusive

chunk=$1

run_id_base="fw10BT_${chunk}"
run_id="${run_id_base}_00${SLURM_ARRAY_TASK_ID}"
model_name="meta-llama/Llama-3.3-70B-Instruct"

DATA_PATH_BASE="../results/fineweb-10BT"
INPUT_FILE="${DATA_PATH_BASE}/fw10BT_${chunk}_00${SLURM_ARRAY_TASK_ID}/descriptors_fw10BT_${chunk}_00${SLURM_ARRAY_TASK_ID}.jsonl"
SCHEMA_FILE="../results/final_schema/descriptor_schema.jsonl"


echo "Running with run_id: $run_id"
echo "Using model: $model_name"
echo "Data path: $INPUT_FILE"

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

# Wait a bit to avoid all jobs requesting model from HF at once.
wait_time=$((SLURM_ARRAY_TASK_ID * 60))
echo "Sleeping for $wait_time seconds to stagger model loading..."
sleep $wait_time

srun singularity run --rocm --bind /scratch/project_462000963 \
    $SIF bash -c "source ../.aif-venv/bin/activate && python harmonize_with_schema.py \
    --run-id=${run_id}  \
    --input=${INPUT_FILE} \
    --schema=${SCHEMA_FILE} \
    --cache-db=../results/harmonized/${run_id}/decision_cache.db \
    --topk=10 \
    --min-embed-score=0.5 \
    --embedder=stella \
    --schema-embed-path=../results/final_schema/schema_embeddings.npz \
    "


PYTHON_EXIT_CODE=$?

# Check the exit code
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    # On success, move results to final location
    echo Moving results to final location...
    mkdir -p ../results/harmonized/fineweb-10BT/${run_id}
    mv "../results/harmonized/${run_id}" ../results/harmonized/fineweb-10BT/${run_id}/
else
    echo "Python script failed with exit code: $PYTHON_EXIT_CODE."
    exit $PYTHON_EXIT_CODE
fi
