#!/bin/bash
#SBATCH --job-name=faiss
#SBATCH --account=project_2017843
#SBATCH --partition=gpumedium
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:2
#SBATCH --mem=320G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

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

descriptor_type=$1

# Use like this.
# python LLM_as_judge.py [global options] <task> [task options]

# possible tasks:
# QueryDescriptorMatch: Judge whether descriptors correspond to a query.
# QueryDocMatch: Judge whether the retrieved documents for a given query are relevant to the query.
# DescriptorAccuracy: Evaluate descriptor accuracy on a sample of documents


srun python LLM_as_judge.py \
                            --model=Qwen/Qwen3-Next-80B-A3B-Instruct \
                            --data-path=../data/weborganizer/topic_format_edu.jsonl\
                            --output-path=../results/aspect_coverage/LLM_judgments_${descriptor_type}.jsonl \
                            --detailed-output \
                            AspectCoverage \
                            --sample-percent 0.05 \
                            --descriptor-type ${descriptor_type} \
