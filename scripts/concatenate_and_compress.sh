#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# Enable nullglob to avoid literal glob patterns when no matches exist
shopt -s nullglob

langs=("srp_Cyrl")

# !NB: This will delete extra files in each directory before concatenation.
# Set to false to skip deletion and keep all files.
DELETE_EXTRA_FILES=false
extra_files=($path_to_dir/descriptors_${lang}_sample_*/decision_cache.db $path_to_dir/descriptors_${lang}_sample_*/*input_embeds.npz)

CONCATENATE=false
COMPRESS=true

path_base="/scratch/project_462000963/users/tarkkaot/LLM_document_descriptors/results/harmonized/HPLT4/Qwen3.5"

for lang in "${langs[@]}"; do
    echo "Processing language: $lang"
    path_to_dir="$path_base/${lang}_sample"
    if [ "$DELETE_EXTRA_FILES" = true ]; then
        echo "Deleting extra files in: $path_to_dir"
        for f in "${extra_files[@]}"; do
            if [[ -f "$f" ]]; then
                echo "Removing: $f"
                rm "$f"
            fi
        done
    fi
    if [ "$CONCATENATE" = true ]; then
        echo "Concatenating..."
        srun bash concatenate_files.sh "$path_to_dir"
    fi
    if [ "$COMPRESS" = true ]; then
        echo "Compressing..."
        srun bash compress_files.sh "$path_to_dir"
    fi
done
