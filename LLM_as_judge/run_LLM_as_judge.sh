#!/bin/bash
#SBATCH --job-name=LLM_judge
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --exclusive

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/scratch/project_462000963/users/tarkkaot/containers/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

# This fixes RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# Set this to avoid errors.
export TORCH_COMPILE_DISABLE=1

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

data_type=$1

# Use like this.
# python LLM_as_judge.py [global options] <task> [task options]

# possible tasks:
# query_correspondence: Judge whether the retrieved documents for a given query are relevant to the query.
# descriptor_accuracy: Evaluate descriptor accuracy on a sample of documents

#example use:
#python LLM_as_judge.py \
#--model='Qwen/Qwen3-Next-80B-A3B-Instruct' \
#sarcasm_query \
#--doc-ids-path=../results/faiss/valid_sarcasm_docs_${data_type}.txt

srun singularity run --rocm --bind /scratch/project_462000963 \
    $SIF bash -c "source ../.aif-venv/bin/activate && python LLM_as_judge.py \
                            --model=Qwen/Qwen3-Next-80B-A3B-Instruct \
                            query_correspondence \
                            --query='sarcasm; the document contains sarcastic humor and irony' \
                            --doc-ids-path=../results/faiss/valid_sarcasm_docs_${data_type}.txt"
