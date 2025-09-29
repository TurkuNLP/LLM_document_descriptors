import argparse
from pathlib import Path
from typing import Iterator, List, Tuple
from multiprocessing import Pool, cpu_count
import zstandard as zstd #type:ignore
import io


def iter_lines(path: Path) -> Iterator[str]:
    """Yield non-empty lines from .jsonl or .jsonl.zst as UTF-8 text."""
    suffixes = path.suffixes
    if suffixes[-2:] == ['.jsonl', '.zst']:
        with path.open('rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    line = line.rstrip('\n')
                    if line.strip():
                        yield line
    elif path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.strip():
                    yield line
    else:
        raise ValueError(f"Unsupported file type: {path.name}. Must be .jsonl or .jsonl.zst")


def count_lines(path: Path) -> int:
    return sum(1 for _ in iter_lines(path))


def base_name_for(path: Path) -> str:
    """Return filename without .jsonl or .jsonl.zst."""
    if path.suffixes[-2:] == ['.jsonl', '.zst']:
        return path.name[:-len('.jsonl.zst')]
    elif path.suffix == '.jsonl':
        return path.stem
    else:
        return path.stem  # fallback


def process_jsonl_file(filepath: Path, output_dir: Path, split_count: int):
    print(f"Processing {filepath.name}...", flush=True)

    total = count_lines(filepath)
    if total == 0:
        print(f"Skipping empty file: {filepath.name}", flush=True)
        return

    # Compute (nearly) equal target sizes per part, skipping empties if total < split_count
    q, r = divmod(total, split_count)
    targets: List[int] = [(q + 1 if i < r else q) for i in range(split_count)]
    targets = [t for t in targets if t > 0]  # produce up to N non-empty parts

    base = base_name_for(filepath)
    it = iter_lines(filepath)

    part_idx = 0
    written_in_part = 0
    current_target = targets[part_idx]

    part_path = output_dir / f"{base}_{part_idx:03d}.jsonl"
    # ensure we don't accidentally append to an old run
    if part_path.exists():
        part_path.unlink()
    f = part_path.open('w', encoding='utf-8')

    try:
        for line in it:
            f.write(line + '\n')
            written_in_part += 1

            if written_in_part >= current_target:
                f.close()
                part_idx += 1
                if part_idx >= len(targets):
                    break  # done
                written_in_part = 0
                current_target = targets[part_idx]
                part_path = output_dir / f"{base}_{part_idx:03d}.jsonl"
                if part_path.exists():
                    part_path.unlink()
                f = part_path.open('w', encoding='utf-8')
        else:
            # exhausted input but last file still open
            f.close()
    finally:
        if not f.closed:
            f.close()

    print(f"Finished splitting {filepath.name} into {part_idx} part(s).", flush=True)


def process_jsonl_file_mp(args: Tuple[Path, Path, int]):
    filepath, output_dir, split_count = args
    process_jsonl_file(filepath, output_dir, split_count)


def split_jsonl_files(input: str, output_dir: str, split_count: int):
    input_path = Path(input)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    if input_path.is_dir():
        for file in input_path.iterdir():
            if file.suffix == '.jsonl' or file.suffixes[-2:] == ['.jsonl', '.zst']:
                jobs.append((file, output_path, split_count))
    elif input_path.is_file():
        if input_path.suffix == '.jsonl' or input_path.suffixes[-2:] == ['.jsonl', '.zst']:
            jobs.append((input_path, output_path, split_count))

    if not jobs:
        print("No .jsonl files found.")
        return

    workers = min(cpu_count(), len(jobs))
    print(f"Splitting {len(jobs)} file(s) using {workers} worker(s)...", flush=True)

    with Pool(processes=workers) as pool:
        pool.map(process_jsonl_file_mp, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL files into N parts.")
    parser.add_argument('--input', required=True, help='Directory containing input files.')
    parser.add_argument('--output-dir', required=True, help='Directory to save split files.')
    parser.add_argument('--split-count', type=int, default=10, help='How many smaller files input files will be split into.')
    args = parser.parse_args()

    split_jsonl_files(args.input, args.output_dir, args.split_count)
