#!/bin/bash
#SBATCH --job-name=embedder_calibration
#SBATCH --account=project_462000353
#SBATCH --partition=dev-g
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

gpu-energy --save

srun python3 calibrate_embedder.py 
gpu-energy --diff