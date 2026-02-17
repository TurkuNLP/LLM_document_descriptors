#!/bin/bash
#SBATCH --job-name=gen_descriptors
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

source ../.venv_pt2.5/bin/activate

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

run_id="dclm_80k"

srun python3 generate_descriptors.py --run-id=$run_id \
                                     --temperature=0.1 \
                                     --batch-size=500 \
                                     --num-batches=40 \
                                     --num-rewrites=1 \
                                     --start-index=80000 \
                                     --data-source="mlfoundations/dclm-baseline-1.0" \
                                     #--text-column="comment_text"

gpu-energy --diff
