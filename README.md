# LLM document descriptors

Repository for the GreenNLP/OpenEuroLLM document descriptor research project. The project uses large language models to create a dynamic taxonomy of descriptive labels ("descriptors") for web documents. Descriptors can be used to filter, subsample, and retrieve relevant documents from a much larger collection.

The code was developed for the LUMI supercomputer ([LUMI](https://lumi-supercomputer.eu/)) and is most useful as a research artifact and starting point for further experiments. It is not a turnkey, platform-independent package.

Interactive demo: <http://159.69.54.25:5000/>. The demo is external to this repository and may no longer be available.

Data release on Hugging Face: <https://huggingface.co/datasets/TurkuNLP/WebDocumentDescriptors>.

## Repository map

- [`descriptor_generation/`](descriptor_generation/) generates descriptors from documents.
- [`disambiguation/`](disambiguation/) groups and disambiguates descriptors.
- [`merging/`](merging/) merges synonymous descriptors and resolves remaining duplicates.
- [`harmonize/`](harmonize/) aligns descriptors with an existing schema.
- [`faiss/`](faiss/) builds descriptor indexes and runs retrieval and LLM-based evaluation.
- [`LLM_as_judge/`](LLM_as_judge/) contains the judging and label-inference pipelines.
- [`eval_descriptors/`](eval_descriptors/) contains descriptor-quality and growth analyses.
- [`notebooks/`](notebooks/) contains exploratory analyses; notebooks may require paths and outputs that are not committed.
- [`scripts/`](scripts/) contains data preparation and utility scripts.
- [`requirements.txt`](requirements.txt) records one Python environment used by the project.

## Workflows

### Generate a new schema

The basic workflow is:

1. Generate descriptors with [`descriptor_generation/generate_descriptors.py`](descriptor_generation/generate_descriptors.py).
2. Extract descriptor groups with [`disambiguation/extract_descriptor_groups.py`](disambiguation/extract_descriptor_groups.py). Use `--num-splits` to divide large inputs for parallel processing.
3. Disambiguate the groups with [`disambiguation/disambiguate_descriptors.py`](disambiguation/disambiguate_descriptors.py).
4. If jobs were run in parallel, concatenate their outputs with [`disambiguation/concat_disambig_results.sh`](disambiguation/concat_disambig_results.sh).
5. Repeat extraction and disambiguation as needed.
6. Merge synonyms with [`merging/merge_synonyms.py`](merging/merge_synonyms.py).
7. If duplicates remain, force-merge them with [`merging/force_merge.py`](merging/force_merge.py).

The `run_*.sh` files beside the Python modules contain LUMI/Slurm examples and should be read before submitting a job.

### Harmonize documents with an existing schema

1. Generate descriptors with [`descriptor_generation/generate_descriptors.py`](descriptor_generation/generate_descriptors.py).
2. Align them with the schema using [`harmonize/harmonize_with_schema.py`](harmonize/harmonize_with_schema.py).

### Run descriptor retrieval

[`faiss/full_pipeline.py`](faiss/full_pipeline.py) runs the FAISS search, judges descriptor matches, evaluates surviving documents, and writes JSONL and summary artifacts. The supplied Slurm wrapper is [`faiss/run_full_pipeline.sh`](faiss/run_full_pipeline.sh); it currently expects `format` or `topic` and `raw` or `harmonized` arguments and contains site-specific paths that must be adapted.

For a new environment, inspect the argument parser in `faiss/full_pipeline.py` and the lower-level scripts in [`faiss/`](faiss/) before adapting the command. The previous README example referred to a non-existent `faiss/run_full_pipeline.py` entry point.

## Running on LUMI

The following is a starting point for the original LUMI setup. Cluster modules, project accounts, partitions, model paths, and Slurm resources are site-specific.

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

Keep model and dataset caches out of the home directory, for example:

```bash
export HF_HOME="/scratch/<your-project>/my_cache"
```

Before submitting a job, update the `--account`, paths, and resource settings in the relevant `run_*.sh` file. Descriptor generation is expensive: `--num-rewrites=0` is faster, while additional rewrites may improve results. The timings in the original project notes were approximately 10 minutes for 500 documents with zero rewrites and one hour with three rewrites, but they depend on the model, node, and workload.

## Reproducibility notes

- The repository targets LUMI, ROCm, and GPU execution. Porting it elsewhere may require changes to the environment and Slurm wrappers.
- The repository relies on a pre-built LUMI AI Factory container image and a virtual environment extension. The container image is not included in the repository. To reproduce the original environment, follow these steps:
1. Load modules
```shell
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

2. Activate container and build venv
```shell
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
singularity shell $SIF
Singularity> python -m venv .venv --system-site-packages
Singularity> source .venv/bin/activate
(.venv) Singularity> pip install -r requirements.txt
```

lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

3. Run scripts using the container and venv in a SLURM job
```shell
srun singularity run --rocm --bind /scratch/project_465002530 \
    $SIF bash -c "source .venv/bin/activate && python script.py \
							  --input 'data.jsonl' \
							  --output 'out.jsonl'
							  "
```

## Citation and license

If you find this repository useful in your research, please cite the following work:

```text
@inproceedings{
tarkkaetal-descriptors,
title={Task-Agnostic Web Document Annotation with LLM-Generated Descriptors},
author={Tarkka, Otto and Henriksson, Erik and Kanerva, Jenna and Ginter, Filip},
year={2026 (forthcoming)},
}
```

