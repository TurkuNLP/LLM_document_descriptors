#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=320G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

search_name=$1

if [ -z "$search_name" ]; then
    echo "Error: Search name is not provided."
    exit 1
fi

cache_dir=$HF_HUB_CACHE
if [ -z "$cache_dir" ]; then
    echo "Error: HF_HUB_CACHE environment variable is not set."
    exit 1
fi

data_type=$2
if [ "$data_type" != "raw" ] && [ "$data_type" != "harmonized" ]; then
    echo "Usage: $0 [raw|harmonized]"
    exit 1
fi


if [ "$data_type" == "raw" ]; then
    index_path="../results/faiss/raw_index.faiss"
    embeddings_path="../results/faiss/raw_embeddings.npy"
    output_path="../results/faiss/raw_search_results_${search_name}.jsonl"
    descriptor_type="raw"
    echo "Running search on raw descriptors..."
elif [ "$data_type" == "harmonized" ]; then
    index_path="../results/faiss/harmonized_index.faiss"
    embeddings_path="../results/faiss/harmonized_embeddings.npy"
    output_path="../results/faiss/harmonized_search_results_${search_name}.jsonl"
    descriptor_type="harmonized"
    echo "Running search on harmonized descriptors..."
fi

srun python search.py \
    --data-path "../results/harmonized/fineweb-edu/concatenated/descriptors_fineweb-edu_harmonized.jsonl" \
    --cache-dir "$cache_dir" \
    --index-path "$index_path" \
    --embeddings-path "$embeddings_path" \
    --descriptor-type "$descriptor_type" \
    --query "legal notices; Contains for example terms of service, legal disclaimers, privacy policies or license agreements" \
    --top-k 100 \
    --max-distance 320 \
    --output-path "$output_path" \
    --build-index \