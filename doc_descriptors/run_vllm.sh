#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
source ../venv/bin/activate

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

run_id="final_zero_vocab"

srun python3 vllm_document_descriptors.py --run-id=$run_id \
                                          --temperature=0.1 \
                                          --batch-size=200 \
                                          --num-batches=20 \
                                          --num-rewrites=3 \
                                          --start-index=16000 \
                                          --max-vocab=0 \
                                          --synonym-threshold=0.3 \
                                          --data-source="fineweb" \
                                          --checkpoint-interval=20 \

gpu-energy --diff
