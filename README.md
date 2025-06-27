Repo for the GreenNLP/OpenEuroLLM document descriptor research project.
The research project aims to use LLMs to create a dynamic taxonomy of descriptive labels ("descriptors") for web documents.

Made to run on the LUMI supercomputer: https://lumi-supercomputer.eu/.
Runs vLLM 0.6.6.

The currently most up-to-date version of the descriptor generation script is `doc_descriptors/doc_descriptors_with_explainers.py`.
If you wish to use the older version that integrates synonym merging into the loop, you can use `doc_descriptors/vllm_document_descriptors.py`. The problem with this version is that the synonym merging doesn't really work, so we've separated it from the descriptor generation loop.

Separate synonym merging scripts can be found in `descriptor_merging`. These are very much work-in-progress as they do not work as intended quite yet.

Below are detailed instructions for running the pipeline on LUMI. If you run this on another machine or cluster, these instructions might not be fully accurate.

To run descriptor generation pipeline on LUMI:

1. Clone this repo into your project `scratch/`

2. `cd doc_descriptors`

3. Create a virtual environment. Read more: https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules
```
module purge
module use /appl/local/csc/modulefiles
module load pytorch
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

4. Install requirements: `pip install -r requirements.txt`

5. By default, the model will be downloaded into you home directory, which will fill up very quickly. I recommend creating a cache folder in your `scratch` and adding this line to your .bashrc so you don't have to worry about setting the cache folder manually: `export HF_HOME="/scratch/<your-project>/my_cache"`. You can also set the caching directory in run_vllm.sh with the flag `--cache-dir`, e.g. `--cache-dir="/scratch/<your-project>/my_cache"`

6. `cd ../scripts`

7. In run_vllm.sh, change `--account` to your project. It is recommended to reserve a full node, i.e., 8 GPUs because reserving less tends to cause NCCL errors. You have to give a `--run-id`, e.g. 'run1'. All other parameters are set to reasonable defaults that you can change if you want to.

8. Run the descriptor generation pipeline: `sbatch run_vllm.sh`
