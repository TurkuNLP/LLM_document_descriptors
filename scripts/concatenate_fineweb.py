from pathlib import Path
import json
import multiprocessing as mp
from functools import partial
from typing import Iterator, Dict, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileTask:
    input_path: Path
    batch_id: int

def process_file(task: FileTask, output_dir: Path) -> None:
    """Process a single input file and write to batch output."""
    output_path = output_dir / f"fineweb-10BT_descriptors_{task.batch_id}.jsonl"
    buffer = []

    try:
        with task.input_path.open('r') as f:
            for line in f:
                data = json.loads(line)
                out_data = {
                    "text": data["text"],
                    "doc_id": data["doc_id"],
                    "unharmonized_descriptors": data["descriptors"][0],
                    "harmonized_descriptors": data["harmonized_descriptors"],
                }
                buffer.append(json.dumps(out_data) + "\n")

                # Write in chunks to reduce I/O operations
                if len(buffer) >= 1000:
                    with output_path.open('a') as wf:
                        wf.writelines(buffer)
                    buffer.clear()

        # Write remaining items
        if buffer:
            with output_path.open('a') as wf:
                wf.writelines(buffer)

    except Exception as e:
        logger.error(f"Failed processing {task.input_path}: {str(e)}")
        raise

def find_input_files(base_dir: Path) -> Iterator[FileTask]:
    """Find all input files and assign to batches."""
    for directory in base_dir.iterdir():
        if not directory.is_dir():
            continue

        try:
            batch_id = int(directory.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue

            for file in subdir.iterdir():
                if file.name.startswith("fw10BT") and file.suffix == ".jsonl":
                    yield FileTask(input_path=file, batch_id=batch_id)

def main():
    base_dir = Path("../results/harmonized/fineweb-10BT")
    output_dir = base_dir  # Or specify different output directory

    # Clear existing output files
    for i in range(15):
        output_path = output_dir / f"fineweb-10BT_descriptors_{i}.jsonl"
        if output_path.exists():
            output_path.unlink()

    # Create process pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Create partial function with fixed output_dir
        worker = partial(process_file, output_dir=output_dir)

        # Process files in parallel
        tasks = list(find_input_files(base_dir))
        logger.info(f"Found {len(tasks)} files to process")

        for _ in pool.imap_unordered(worker, tasks, chunksize=10):
            pass  # Progress is logged in worker

if __name__ == "__main__":
    main()