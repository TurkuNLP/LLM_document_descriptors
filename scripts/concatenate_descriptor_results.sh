#!/usr/bin/env bash
# Usage: ./gather_descriptors.sh /path/to/main-directory [name_pattern] [output_filename]
# Defaults: name_pattern="descriptors_new*"  output_filename="descriptors_new.jsonl"

set -euo pipefail

main_dir="${1:-.}"
pattern="${2:-descriptors_new*}"
output="${3:-descriptors_new.jsonl}"

# Resolve absolute path
main_dir="$(cd "$main_dir" && pwd)"
outpath="$main_dir/$output"

# Start fresh
: > "$outpath"

# Find matching files exactly one level below main_dir, concatenate in sorted order
while IFS= read -r -d '' file; do
  cat "$file" >> "$outpath"
    # Add a newline only if the file's last byte isn't a newline
    if [ -s "$file" ] && [ "$(tail -c 1 "$file" 2>/dev/null)" != $'\n' ]; then
    printf '\n' >> "$outpath"
    fi
done < <(
  find "$main_dir" -mindepth 2 -maxdepth 2 -type f -name "$pattern" -print0 \
  | sort -z
)

echo "Wrote: $outpath"
