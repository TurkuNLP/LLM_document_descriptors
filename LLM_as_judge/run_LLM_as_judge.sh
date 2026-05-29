#!/bin/bash
#SBATCH --job-name=LLM_judge
#SBATCH --account=project_462000964
#SBATCH --partition=standard-g
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=80G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --exclusive

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/scratch/project_462000963/users/tarkkaot/containers/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

# This fixes RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# Set this to avoid errors.
export TORCH_COMPILE_DISABLE=1

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

CLI=$1

# Use like this.
# python LLM_as_judge.py [global options] <task> [task options]

# possible tasks:
# QueryDescriptorMatch: Judge whether descriptors correspond to a query.
# QueryDocMatch: Judge whether the retrieved documents for a given query are relevant to the query.
# DescriptorAccuracy: Evaluate descriptor accuracy on a sample of documents

path_base=../data/query_samples
data_paths=${path_base}/validated_all_${CLI}_harmonized.jsonl

output_base=../results/LLM_as_judge
output_paths=${output_base}/QueryDocMatch_validated_all_${CLI}_harmonized.jsonl


sarcasm_query='sarcasm; the document contains sarcastic humor and irony'
legal_query='legal notices; Contains for example terms of service, legal disclaimers, privacy policies or license agreements'
faq_query='faq; The page content is in the Frequently Asked Questions format'

if [ $CLI == "sarcasm" ]; then
    query=$sarcasm_query
elif [ $CLI == "legal" ]; then
    query=$legal_query
elif [ $CLI == "FAQ" ]; then
    query=$faq_query
else
    echo "Unknown CLI argument: $CLI. Expected 'sarcasm', 'legal' or 'FAQ'."
    exit 1
fi

echo "Running LLM_as_judge with query: $query"
echo "Data paths: $data_paths"
echo "Output paths: $output_paths"

srun singularity run --rocm --bind /scratch/project_462000963 \
    $SIF bash -c "source ../.aif-venv/bin/activate && python LLM_as_judge.py \
                            --model=Qwen/Qwen3-Next-80B-A3B-Instruct \
                            --data-path=$data_paths \
                            --output-path=$output_paths \
                            --detailed-output \
                            QueryDocMatch \
                            --query='$query' \
                            "
