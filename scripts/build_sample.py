import io
import random
import zstandard as zstd  # type: ignore
from pathlib import Path
from typing import List, Iterator
import argparse
import time

# This script samples documents from JSONL files in a directory, with support for .zst compressed files.
# It uses an initial random sampling step to keep a fraction of the documents,
# and then performs a final random sample to get the desired number of samples.
# The sample will not be perfectly uniform due to the initial sampling step,
# but it allows for efficient sampling from large datasets.


def load_jsonl_file(file_path: str) -> Iterator[dict]:
    """Load JSONL file, decompressing if it's .zst format."""
    if file_path.endswith(".zst"):
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                    for line in text_reader:
                        if line.strip():
                            yield line
    else:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    yield line


def get_jsonl_files(input_path: str) -> List[str]:
    """Get list of JSONL files from directory or single file."""
    path = Path(input_path)

    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl*"), key=lambda f: f.name)
        return [str(f) for f in files if f.is_file()]
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")


def save_checkpoint(sampled: List[str], checkpoint_path: str):
    """Save intermediate sample to checkpoint file."""
    with open(checkpoint_path, "a") as f:
        for example in sampled:
            f.write(example if example.endswith("\n") else example + "\n")


def sample_data(
    input_path: str,
    output_path: str,
    sample_size: int,
    initial_keep_prob: float,
    checkpoint_interval: int,
) -> None:

    # Collect initial sample
    sampled = []

    line_counter = 0
    doc_counter = 0
    num_sampled = 0

    start_time = time.time()
    # Input can be a directory containing multiple JSONL files or a single JSONL file
    # If compressed files are present, they should have .zst extension and will be decompressed on the fly
    files_to_process = get_jsonl_files(input_path)
    for file_path in files_to_process:
        doc_counter += 1
        for line in load_jsonl_file(file_path):
            line_counter += 1
            if (
                random.random() <= initial_keep_prob
            ):  # keep initial_keep_prob of the documents in the initial sample
                num_sampled += 1
                sampled.append(line)

        end_time = time.time()
        print(
            f"{doc_counter}: Read {line_counter:,} lines,"
            f"sampled {num_sampled:,} documents in {end_time - start_time:.2f} seconds",
            flush=True,
        )

        if doc_counter % checkpoint_interval == 0:
            print(
                f"Processed {doc_counter}/{len(files_to_process)} files."
                f" Expected initial sample size: {int(len(files_to_process) * (num_sampled/doc_counter)):,} documents",
                flush=True,
            )
            print("Saving checkpoint...", flush=True)
            save_checkpoint(sampled, f"{output_path}.checkpoint")
            sampled = (
                []
            )  # Clear the sampled list to free memory after saving checkpoint

    if sampled:  # Save any remaining samples after processing all files
        save_checkpoint(sampled, f"{output_path}.checkpoint")
        sampled = []

    print("Finished initial sampling.", flush=True)
    # Load the initial sample from the checkpoint file
    with open(f"{output_path}.checkpoint", "r") as f:
        sampled = [line for line in f if line.strip()]

    print(f"{len(sampled):,} documents in the initial sample", flush=True)

    # Get final sample of the desired size from the initial sample
    # If the initial sample is smaller than the desired sample size, we just use all of it
    final_sample = (
        random.sample(sampled, sample_size) if len(sampled) > sample_size else sampled
    )

    print(f"{len(final_sample):,} documents in the final sample", flush=True)

    with open(output_path, "w") as f:
        for example in final_sample:
            f.write(example if example.endswith("\n") else example + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sample JSONL files using reservoir sampling"
    )
    parser.add_argument("--input", help="Input directory or file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "-n", type=int, default=1000, help="Number of samples (default: 1000)"
    )
    parser.add_argument(
        "--initial-keep-prob",
        type=float,
        default=0.01,
        help="Initial probability to keep a document (default: 0.01)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save intermediate checkpoint every N documents (default: 10)",
    )

    args = parser.parse_args()
    sample_data(
        args.input,
        args.output,
        args.n,
        args.initial_keep_prob,
        args.checkpoint_interval,
    )
