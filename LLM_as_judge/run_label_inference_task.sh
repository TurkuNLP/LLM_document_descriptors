#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_2017843
#SBATCH --partition=gpumedium
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:2
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

module load python-vllm/0.18.0
source ../.vllm0.18_venv/bin/activate

export HF_HOME="/scratch/project_2017843/tarkkaot/hf_home"
cache_dir=$HF_HOME

label_type=$1
descriptor_type=$2

if [ "$label_type" != "format" ] && [ "$label_type" != "topic" ]; then
    echo "Usage: $0 [format|topic] [raw|harmonized]"
    exit 1
fi

if [ "$descriptor_type" != "raw" ] && [ "$descriptor_type" != "harmonized" ]; then
    echo "Usage: $0 [format|topic] [raw|harmonized]"
    exit 1
fi

#if [ "$label_type" == "format" ]; then
#    data_path="../data/weborganizer/LLM_verified_formats_edu.jsonl"
#elif [ "$label_type" == "topic" ]; then
#    data_path="../data/weborganizer/LLM_verified_topics_edu.jsonl"
#fi

data_path="../data/weborganizer/topic_format_edu.jsonl"

output_dir="../results/LLM_as_judge/label_inference_task/"

mkdir -p "$output_dir"

output_path="$output_dir/inferred_${label_type}_labels_${descriptor_type}_descriptors.jsonl"


echo "Data path: $data_path"
echo "Output path: $output_path"

srun python LLM_as_judge.py \
    --model=Qwen/Qwen3-Next-80B-A3B-Instruct \
    --cache-dir=$cache_dir \
    --data-path=$data_path \
    --output-path=$output_path \
    --detailed-output \
    InferLabelsFromDescriptors \
    --label-type=$label_type \
    --descriptor-type=$descriptor_type \
                            