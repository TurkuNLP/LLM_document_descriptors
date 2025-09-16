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

main_dir="${1:-.}"
pattern="descriptors_new*"
out="${main_dir%/}/all_descriptors_new.jsonl"

# Create output file if it doesn't exist (do not truncate)
touch "$out"

# Iterate immediate subdirectories of main_dir
for dir in "$main_dir"/*/ ; do
  [[ -d "$dir" ]] || continue

  # For each file starting with "descriptors_new" in this subdirectory
  for f in "$dir"/$pattern ; do
    [[ -f "$f" ]] || continue
    echo "Appending: $f"
    # Print each line and ensure a final newline (important for JSONL merges)
    awk '1' "$f" >> "$out"
  done
done

echo "Done. Appended matches into: $out"
echo "Total lines in $out: $(wc -l < "$out")"
