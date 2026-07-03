#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_2017843
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
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
module load python-vllm/0.18.0
source ../.vllm0.18_venv/bin/activate

# Read queries from WO_formats.txt and concatenate with |
query_file="WO_formats.txt"
if [ ! -f "$query_file" ]; then
    echo "Error: Query file $query_file not found."
    exit 1
fi

# Read file, trim whitespace, filter empty lines, join with |
query=$(awk '{$1=$1};1' "$query_file" | grep -v '^$' | paste -sd "|" -)
if [ -z "$query" ]; then
    echo "Error: No valid queries found in $query_file"
    exit 1
fi

echo "Using concatenated query: $query"

cache_dir=$HF_HOME
if [ -z "$cache_dir" ]; then
    echo "Error: HF_HOME environment variable is not set."
    exit 1
fi

label_type=$1
if [ "$label_type" != "format" ] && [ "$label_type" != "topic" ]; then
    echo "Usage: $0 [format|topic] [raw|harmonized]"
    exit 1
fi

data_type=$2
if [ "$data_type" != "raw" ] && [ "$data_type" != "harmonized" ]; then
    echo "Usage: $0 [format|topic] [raw|harmonized]"
    exit 1
fi


if [ "$data_type" == "raw" ] && [ "$label_type" == "format" ]; then
    data_path="../data/weborganizer/LLM_verified_formats_edu.jsonl"
    index_path="../results/faiss/format_raw_index.faiss"
    embeddings_path="../results/faiss/format_raw_embeddings.npy"
    output_dir="../results/faiss/pipeline/formats/raw"
    descriptor_type="raw"
    echo "Running format search on raw descriptors..."
elif [ "$data_type" == "raw" ] && [ "$label_type" == "topic" ]; then
    data_path="../data/weborganizer/LLM_verified_topics_edu.jsonl"
    index_path="../results/faiss/topic_raw_index.faiss"
    embeddings_path="../results/faiss/topic_raw_embeddings.npy"
    output_dir="../results/faiss/pipeline/topics/raw"
    descriptor_type="raw"
    echo "Running topic search on raw descriptors..."
elif [ "$data_type" == "harmonized" ] && [ "$label_type" == "format" ]; then
    data_path="../data/weborganizer/LLM_verified_formats_edu.jsonl"
    index_path="../results/faiss/format_harmonized_index.faiss"
    embeddings_path="../results/faiss/format_harmonized_embeddings.npy"
    output_dir="../results/faiss/pipeline/formats/harmonized"
    descriptor_type="harmonized"
    echo "Running format search on harmonized descriptors..."
elif [ "$data_type" == "harmonized" ] && [ "$label_type" == "topic" ]; then
    data_path="../data/weborganizer/LLM_verified_topics_edu.jsonl"
    index_path="../results/faiss/topic_harmonized_index.faiss"
    embeddings_path="../results/faiss/topic_harmonized_embeddings.npy"
    output_dir="../results/faiss/pipeline/topics/harmonized"
    descriptor_type="harmonized"
    echo "Running topic search on harmonized descriptors..."
fi

mkdir -p $output_dir

srun python full_pipeline.py \
    --data-path $data_path \
    --cache-dir $cache_dir \
    --index-path $index_path \
    --embeddings-path $embeddings_path \
    --descriptor-type $descriptor_type \
    --query "$query" \
    --top-k 100 \
    --max-distance 320 \
    --output-dir $output_dir \
    --build-index \