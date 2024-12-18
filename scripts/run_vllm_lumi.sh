#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=4
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --exclude=nid007955

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# activate venv to use sentence_transformers
# since it's not part of the pytorch module
source ../venv/bin/activate

gpu-energy --save

srun python3 vllm_document_descriptors.py --run-id='70B_2' --temperature=0.1

gpu-energy --diff
