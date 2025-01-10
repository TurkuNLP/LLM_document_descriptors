#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=4
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --exclude=nid007955,nid005022,nid007957

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
source ../venv/bin/activate

# Apparently some hipster library likes to fill your home folder with cache, so put it in scratch instead.
TRITON_HOME=/scratch/project_462000353/tarkkaot/LLM_document_descriptors/hf_cache/

gpu-energy --save

srun python3 vllm_document_descriptors.py --run-id='70B_3.3_5' --temperature=0.1 --batch-size=50 --num-rewrites=3 --start-index=5000 --num-batches=20

gpu-energy --diff
