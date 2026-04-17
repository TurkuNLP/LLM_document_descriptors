#!/usr/bin/env bash
# concat_descriptors.sh
# Usage:
#   ./concat_descriptors.sh [MAIN_DIR]
# - MAIN_DIR defaults to current directory.
# - Looks only one level deep (immediate subdirectories).
# - Appends (>>) matching files to MAIN_DIR/all_descriptors_new.jsonl.
# - Ensures each source file ends with a newline (good for JSONL).

set -euo pipefail
shopt -s nullglob

# Main directory to look for files (default: current)
path_to_dir="${1:-.}"

# Get name of main directory
main_dir_name="$(basename "$path_to_dir")"
echo "Main directory: $path_to_dir"
# File name pattern to match. E.g., "merge_array_ids_*.jsonl"
pattern="descriptors_*.jsonl"
#pattern="full_lineage.jsonl"

# Output file path
out="${path_to_dir%/}/../all/${main_dir_name}_harmonized.jsonl"
#out="${path_to_dir%/}/all_merges_full_lineage.jsonl"

# Create output file if it doesn't exist (do not truncate)
# touch "$out"

# Iterate immediate subdirectories of main_dir
for dir in "$path_to_dir"/*/ ; do
  [[ -d "$dir" ]] || continue

  # For each file starting with $pattern in this subdirectory
  for f in "$dir"/$pattern ; do
    [[ -f "$f" ]] || continue
    echo "Appending: $f"
    # Print each line and ensure a final newline
    awk '1' "$f" >> "$out"
  done
done

echo "Done. Appended matches into: $out"
echo "Total lines in $out: $(wc -l < "$out")"
