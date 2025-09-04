#!/usr/bin/env bash
# count_descriptor_lines.sh
# Usage: ./count_descriptor_lines.sh /path/to/main-directory [name_pattern]
# Defaults: name_pattern="descriptors_new*"
# Prints: "<relative/path>\t<line_count>" in deterministic (sorted) order.

set -euo pipefail

main_dir="${1:-.}"
pattern="${2:-descriptors_new*}"

# Resolve absolute path
main_dir="$(cd "$main_dir" && pwd)"

# Header (optional—comment out if you don't want it)
printf "file\tlines\n"

found=false

# List files exactly one level below main_dir, match pattern, sort for determinism
while IFS= read -r -d '' file; do
  found=true
  # Count lines robustly (counts last line even without trailing newline)
  lines="$(awk 'END{print NR}' "$file")"
  # Print path relative to main_dir
  rel="${file#"$main_dir"/}"
  printf "%s\t%s\n" "$rel" "$lines"
done < <(
  find "$main_dir" -mindepth 2 -maxdepth 2 -type f -name "$pattern" -print0 \
  | sort -z
)

if ! $found; then
  >&2 echo "No files matching \"$pattern\" found one level below: $main_dir"
fi
