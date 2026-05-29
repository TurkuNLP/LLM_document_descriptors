#!/bin/bash
#SBATCH --job-name=LLM_judge
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

set -euo pipefail

module purge
module use /appl/local/csc/modulefiles
module load pytorch

data_type=$1

if [ -z "$data_type" ]; then
  echo "Error: data_type argument is required."
  echo "Usage: $0 <data_type>"
  echo "Example: $0 faq"
  exit 1
fi

# Use like this.
# python LLM_as_judge.py [global options] <task> [task options]

# possible tasks:
# QueryDescriptorMatch: Judge whether descriptors correspond to a query.
# QueryDocMatch: Judge whether the retrieved documents for a given query are relevant to the query.
# DescriptorAccuracy: Evaluate descriptor accuracy on a sample of documents

echo "Running LLM_as_judge with data_type: $data_type"
echo "Using model: $ALIBABA_API_MODEL"
echo "API host: $ALIBABA_API_HOST"

srun python3 LLM_as_judge.py --model=$ALIBABA_API_MODEL \
                            --backend=api \
                            --api-base-url=$ALIBABA_API_HOST \
                            --api-key=$ALIBABA_API_KEY \
                            --data-path=../results/faiss/${data_type}_search_results_faq.jsonl \
                            --output-path=../results/LLM_as_judge/QueryDescriptorMatch_${data_type}_faq.jsonl \
                            --detailed-output \
                            QueryDescriptorMatch \
                            --query="faq; The page content is in the Frequently Asked Questions format"