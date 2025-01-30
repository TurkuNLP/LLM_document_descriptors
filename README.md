Repo for the GreenNLP/HPLT document descriptor research project.
The research project aims to use LLMs to create a dynamic taxonomy of descriptive labels ("descriptors") for internet documents.

Made to run on the LUMI supercomputer: https://lumi-supercomputer.eu/.
Runs vLLM 0.6.6.

To run on LUMI:

1. Create a virtual environment. Read more: https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules
```
cd /projappl/<your_project>  # change this to the appropriate path for your project
module load python-data
python3 -m venv --system-site-packages <venv_name>
source <venv_name>/bin/activate
```


2. In run_vllm.sh, change `--account` to your project. It is recommended to reserve a full node, i.e., 8 GPUs because reserving less tends to cause NCCL errors.

3. Run the descriptor generation pipeline: `sbatch run_vllm.sh`
