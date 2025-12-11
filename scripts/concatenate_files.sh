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
main_dir="${1:-.}"
# File name pattern to match. E.g., "merge_array_ids_*.jsonl"
pattern="merge_array_ids_*.jsonl"
#pattern="full_lineage.jsonl"

# Output file path
out="${main_dir%/}/all_merges_disambig.jsonl"
#out="${main_dir%/}/all_merges_full_lineage.jsonl"

# Create output file if it doesn't exist (do not truncate)
touch "$out"

# Iterate immediate subdirectories of main_dir
for dir in "$main_dir"/*/ ; do
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
