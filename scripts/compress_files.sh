#!/bin/bash
#SBATCH --job-name=compress
#SBATCH --account=project_462000963
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

###############################################################################
# Compress every file in multiple directories with zstd, skipping anything that
# already ends in .zst. Removes originals only after a successful compression.
#
#   sbatch compress_zstd.sbatch  "/path/to/dir1 /path/to/dir2 ..." [compression-level]
#
# Arguments
#   $1...$n-1  Directories to process (required, can use wildcards like dir_*)
#   $n         Optional zstd level (1–22, default 19)
#
# Notes
#   • Each file is compressed **independently** so failures affect only
#     that file.
#   • Both intra-file threading (-T) and inter-file parallelism (xargs -P)
#     are used, taking advantage of all CPUs Slurm allocates.
###############################################################################

set -euo pipefail

# ---- Parse and validate user input -----------------------------------------
# Check if at least one argument is provided
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 dir1 [dir2 dir3 ...] [compression-level]" >&2
  exit 1
fi

# Check if the last argument is numeric (compression level)
if [[ $# -gt 1 ]] && [[ "${!#}" =~ ^[0-9]+$ ]]; then
  LEVEL="${!#}"  # Use the last argument as compression level
  DIRS=("${@:1:$#-1}")  # All arguments except the last one are directories
else
  LEVEL=19  # Default compression level
  DIRS=("$@")  # All arguments are directories
fi

THREADS=${SLURM_CPUS_PER_TASK:-1}   # Use all cores Slurm gives us

# ---- Environment for multithreaded zstd ------------------------------------
export ZSTD_NBTHREADS="$THREADS"    # Honour -T0 (auto) inside zstd

echo "Compressing files in '${DIRS[*]}' with zstd -${LEVEL} using ${THREADS} threads."

# ---- Main compression loop --------------------------------------------------
# Process each directory
for DIR in "${DIRS[@]}"; do
  if [[ ! -d "$DIR" ]]; then
    echo "Warning: '$DIR' is not a directory, skipping." >&2
    continue
  fi

  echo "Processing directory: $DIR"
  
  # • zstd -r: recursive compression of directory
  #   --rm = delete the source file only if compression succeeds
  zstd -r --rm -T0 -"$LEVEL" --exclude-compressed --quiet "$DIR"
  
  # Alternative method with find/xargs (uncomment if needed):
  # find "$DIR" -type f ! -name '*.zst' -print0 |
  #   xargs -0 -n1 -P "$THREADS" zstd -"$LEVEL" -T0 --rm --quiet
done

echo "Done."