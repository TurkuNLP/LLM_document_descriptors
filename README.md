Repo for the GreenNLP/OpenEuroLLM document descriptor research project.
The research project aims to use LLMs to create a dynamic taxonomy of descriptive labels ("descriptors") for web documents.

Made to run on the LUMI supercomputer: https://lumi-supercomputer.eu/.
Runs vLLM 0.6.6.

### Generating a new schema
The basic steps for generating a new descriptor schema from scratch are:
1. Generate a raw set of descriptors with `descriptor_generation/generate_descriptor.py`.
2. Extract descriptor groups from raw results with `disambiguation/extract_descriptor_groups.py`. Split data into smaller batches for faster parallel processing by setting --num-splits.
3. Use the extracted descriptor groups as input for `disambiguation/disambiguate_descriptors.py`.
4. If you ran multiple jobs in parallel (which you should with any larger data), concatenate all results together with `disambiguation/concat_disambig_results.sh`.
5. You'll probably have to repeat this a few times. So extract groups from the output of the disambiguation, disambiguate again, etc. Concatenate results after each disambiguation run if running in parallel.
6. Merge the results of you final disambiguation run with `merging/find_synonyms.py`.
7. If duplicates still remain, force merge them with `merging/force_merge.py`.

You schema is done!

### Generating new descriptor for existing schema
Once you have a schema, you can generate descriptors for any dataset and then align those descriptors with the schema.

1. Generate a raw set of descriptors with `descriptor_generation/generate_descriptor.py`.
2. Harmonize descriptors with your schema with `harmonize/harmonize_with_schema.py`.

Harmonization done!

### More details

Below are more detailed instructions for running the pipeline on LUMI. If you run this on another machine or cluster, these instructions might not be fully accurate.

To run descriptor generation pipeline on LUMI:

1. Clone this repo into your project `scratch/`

2. `cd doc_descriptors`

3. Create a virtual environment. Read more: https://docs.csc.fi/support/tutorials/python-usage-guide/#installing-python-packages-to-existing-modules
```
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

4. Install requirements: `pip install -r requirements.txt`

5. By default, the model will be downloaded into you home directory, which will fill up very quickly. I recommend creating a cache folder in your `scratch` and adding this line to your .bashrc so you don't have to worry about setting the cache folder manually: `export HF_HOME="/scratch/<your-project>/my_cache"`. You can also set the caching directory in run_vllm.sh with the flag `--cache-dir`, e.g. `--cache-dir="/scratch/<your-project>/my_cache"`

7. In `run_generate_descriptors.sh`, change `--account` to your project. It is recommended to reserve a full node, i.e., 8 GPUs because reserving less tends to cause NCCL errors. You have to give a `--run-id`, e.g. 'run1'. All other parameters are set to reasonable defaults that you can change if you want to.

8. Run the descriptor generation pipeline: `sbatch run_vllm.sh`
