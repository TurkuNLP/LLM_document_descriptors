#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_2011109
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=320G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


# Run full FAISS search & LLM evaluation pipeline including:
# 1. FAISS search for query.
# 2. LLM-based judgement of descriptor matches.
# 3. LLM-based judgement of document matches.
# Run pipeline for multiple queries by contatenating them with "|".

# Set the number of CPU threads based on cpus-per-task
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
#export OMP_PLACES=cores
#export OMP_PROC_BIND=spread


module purge
module load python-vllm
source ../.vllm_venv/bin/activate

label_type=$1
if [ -z "$label_type" ]; then
    echo "Error: No label type provided. Usage: $0 [label_type] [descriptor_type]"
    exit 1
fi

if [ "$label_type" != "topic" ] && [ "$label_type" != "format" ]; then
    echo "Error: Invalid label type. Please specify either 'topic' or 'format'."
    exit 1
fi

descriptor_type=$2
if [ -z "$descriptor_type" ]; then
    echo "Error: No descriptor type provided. Usage: $0 [label_type] [descriptor_type]"
    exit 1
fi

if [ "$descriptor_type" != "harmonized" ] && [ "$descriptor_type" != "raw" ]; then
    echo "Error: Invalid descriptor type. Please specify either 'harmonized' or 'raw'."
    exit 1
fi

cache_dir=$HF_HOME
output_dir="../results/LLM_as_judge/label2doc_desc2label/${label_type}/${descriptor_type}/"
mkdir -p $output_dir

output_path="${output_dir}desc2label_responses.jsonl"

echo "Running label2doc_desc2label pipeline with label type: $label_type and descriptor type: $descriptor_type"

srun python label2doc_desc2label_pipeline.py \
    --label-type $label_type \
    --descriptor-type $descriptor_type \
    --cache-dir $cache_dir \
    --output-path $output_path \
    --label2document-results-path ../results/LLM_as_judge/label2doc_${label_type}_results.jsonl