#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


# Run full FAISS search & LLM evaluation pipeline including:
# 1. FAISS search for query.
# 2. LLM-based judgement of descriptor matches.
# 3. LLM-based judgement of document matches.
# Run pipeline for multiple queries by contatenating them with "|".


OELLM_CONTAINER="/scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif"

module purge
module use /appl/local/laifs/modules/
module load lumi-aif-singularity-bindings
export SIF=$OELLM_CONTAINER

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

data_type=$1
if [ "$data_type" != "raw" ] && [ "$data_type" != "harmonized" ]; then
    echo "Usage: $0 [raw|harmonized]"
    exit 1
fi


if [ "$data_type" == "raw" ]; then
    index_path="../results/faiss/raw_index.faiss"
    embeddings_path="../results/faiss/raw_embeddings.npy"
    output_dir="../results/faiss/pipeline/formats/raw"
    descriptor_type="raw"
    echo "Running search on raw descriptors..."
elif [ "$data_type" == "harmonized" ]; then
    index_path="../results/faiss/harmonized_index.faiss"
    embeddings_path="../results/faiss/harmonized_embeddings.npy"
    output_dir="../results/faiss/pipeline/formats/harmonized"
    descriptor_type="harmonized"
    echo "Running search on harmonized descriptors..."
fi

export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

srun singularity run --rocm --bind /scratch/project_465002530 \
    $SIF bash -c "source /scratch/project_465002530/users/tarkkaot/LLM_document_descriptors/.laif-venv/bin/activate && python full_pipeline.py \
    --data-path ../results/harmonized/fineweb-edu/concatenated/descriptors_fineweb-edu_harmonized.jsonl \
    --cache-dir '$cache_dir' \
    --index-path '$index_path' \
    --embeddings-path '$embeddings_path' \
    --descriptor-type '$descriptor_type' \
    --query '$query' \
    --top-k 100 \
    --max-distance 320 \
    --output-dir '$output_dir'" \