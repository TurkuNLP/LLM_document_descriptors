#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
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

cache_dir=$HF_HUB_CACHE
if [ -z "$cache_dir" ]; then
    echo "Error: HF_HUB_CACHE environment variable is not set."
    exit 1
fi

data_type=$1
if [ "$data_type" != "raw" ] && [ "$data_type" != "harmonized" ]; then
    echo "Usage: $0 [raw|harmonized]"
    exit 1
fi


if [ "$data_type" == "raw" ]; then
    index_path="../results/faiss/raw_index.faiss"
    embeddings_path="../results/faiss/raw_embeddings.npy"
    output_path="../results/faiss/raw_search_results.jsonl"
    descriptor_type="raw"
    echo "Running search on raw descriptors..."
elif [ "$data_type" == "harmonized" ]; then
    index_path="../results/faiss/harmonized_index.faiss"
    embeddings_path="../results/faiss/harmonized_embeddings.npy"
    output_path="../results/faiss/harmonized_search_results.jsonl"
    descriptor_type="harmonized"
    echo "Running search on harmonized descriptors..."
fi

srun python search.py \
    --data-path "../results/harmonized/fineweb-edu/concatenated/descriptors_fineweb-edu_harmonized.jsonl" \
    --cache-dir "$cache_dir" \
    --index-path "$index_path" \
    --embeddings-path "$embeddings_path" \
    --descriptor-type "$descriptor_type" \
    --query "sarcasm; the document contains sarcastic humor and irony" \
    --top-k 100 \
    --max-distance 250 \
    --output-path "$output_path" \
    --build-index \