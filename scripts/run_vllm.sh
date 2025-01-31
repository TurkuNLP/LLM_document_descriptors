#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
source ../venv/bin/activate

# Apparently some hipster library likes to fill your home folder with cache, so put it in scratch instead.
TRITON_HOME=/scratch/project_462000353/tarkkaot/LLM_document_descriptors/.cache/

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

srun python3 vllm_document_descriptors.py --run-id="70B_3.3_0-test0" \
                                          --temperature=0.1 \
                                          --batch-size=50 \
                                          --num-batches=20 \
                                          --num-rewrites=3 \
                                          --start-index=0 \
                                          --num-batches=20 \
                                          --max-vocab=50 \
                                          #--use-previous-descriptors \
                                          #--descriptor-path="/scratch/project_462000353/tarkkaot/LLM_document_descriptors/results/descriptor_vocab_70B_3.3_0_full-vocab.tsv" \

gpu-energy --diff
