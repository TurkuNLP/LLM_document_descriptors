Repo for the GreenNLP/HPLT document descriptor research project.
The research project aims to use LLMs to create a dynamic taxonomy of descriptive labels ("descriptors") for internet documents.

Made to run on the LUMI supercomputer: https://lumi-supercomputer.eu/.
Runs vLLM 0.6.6.

To run on LUMI:

1. Clone this repo into your project `scratch/`

2. `cd LLM_document_descriptors`

3. Create a virtual environment. Read more: https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules
```
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

4. Install requirements: `pip install -r requirements.txt`

5. In run_vllm.sh, change `--account` to your project. It is recommended to reserve a full node, i.e., 8 GPUs because reserving less tends to cause NCCL errors.

6. Update `load_documents()` in utils.py to load your data.

7. Run the descriptor generation pipeline: `sbatch run_vllm.sh`
