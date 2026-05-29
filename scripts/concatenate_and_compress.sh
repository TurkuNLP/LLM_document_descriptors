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

file_ids=("srp_Cyrl")

# !NB: This will delete extra files in each directory before concatenation.
# Set to false to skip deletion and keep all files.
DELETE_EXTRA_FILES=false
extra_files=($path_to_dir/decision_cache.db $path_to_dir/*input_embeds.npz)

CONCATENATE=false
COMPRESS=true

path_base="/scratch/project_462000963/users/tarkkaot/LLM_document_descriptors/results/harmonized/fineweb-10BT"

find "$path_base" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' dir; do
    echo "Processing directory: $dir"
    extra_files=($dir/decision_cache.db $dir/*input_embeds.npz)
    if [ "$DELETE_EXTRA_FILES" = true ]; then
        echo "Deleting extra files in: $dir"
        for f in "${extra_files[@]}"; do
            if [[ -f "$f" ]]; then
                echo "Removing: $f"
                rm "$f"
            fi
        done
    fi
    if [ "$CONCATENATE" = true ]; then
        echo "Concatenating..."
        find "$dir" -maxdepth 1 -type f -name '*.npz' | xargs -P "$SLURM_CPUS_PER_TASK" -I {} bash concatenate_files.sh {}
    fi
    if [ "$COMPRESS" = true ]; then
        echo "Compressing..."
        srun bash compress_files.sh "$dir"
    fi
done