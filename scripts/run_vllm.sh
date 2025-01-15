#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --exclude=nid005022,nid005023,nid005024,nid007955,nid007956,nid007957

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
source ../venv/bin/activate

# Apparently some hipster library likes to fill your home folder with cache, so put it in scratch instead.
TRITON_HOME=/scratch/project_462000353/tarkkaot/LLM_document_descriptors/.cache/

# Log NCCL info to catch what is causing the NCCL errors.
export NCCL_DEBUG=INFO

gpu-energy --save

srun python3 vllm_document_descriptors.py --run-id='70B_3.3_4_max-vocab-50' \
                                          --temperature=0.1 \
                                          --batch-size=50 \
                                          --num-rewrites=3 \
                                          --start-index=4000 \
                                          --num-batches=20 \
                                          --max-vocab=50

gpu-energy --diff
