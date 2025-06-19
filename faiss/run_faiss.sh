#!/bin/bash
#SBATCH --job-name=faiss_search
#SBATCH --account=project_2011109
#SBATCH --partition=medium
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
###SBATCH --cpus-per-task=96
#SBATCH --mem=0 # Use all available memory
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# === Load Required Modules ===
module purge
#module use /appl/local/csc/modulefiles
module load pytorch

# === Activate Python Virtual Environment ===
source ../venv/bin/activate

###export OMP_NUM_THREADS=96  # Adjust to match the number of CPUs allocated
RUN_ID="$1"

python3 faiss_neighbors.py --run-id "$RUN_ID" \
                           --data-path "/scratch/project_2011109/erik/schema-on-demand/data/processed/descriptors_with_explainers_embeddings.jsonl" \
                           --data-size=-1 \
                           --k=1 \
                           --max-iter=50 \
