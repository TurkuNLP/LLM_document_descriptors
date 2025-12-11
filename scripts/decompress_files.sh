#!/bin/bash
#SBATCH --job-name=decompress
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
# Decompress all zstd-compressed files in multiple directories. Optionally
# removes compressed files after successful decompression.
#
#   sbatch decompress_zstd.sh "/path/to/dir1 /path/to/dir2 ..." [--keep]
#
# Arguments
#   $1...$n  Directories to process (required, can use wildcards like dir_*)
#   --keep   Optional flag to keep original compressed files (default: remove)
#
# Notes
#   • Each file is decompressed **independently** so failures affect only
#     that file.
#   • Both intra-file threading (-T) and inter-file parallelism are used, 
#     taking advantage of all CPUs Slurm allocates.
###############################################################################

set -euo pipefail

# ---- Parse and validate user input -----------------------------------------
# Default settings
KEEP_COMPRESSED=false
DIRS=()

# Process arguments
for arg in "$@"; do
  if [[ "$arg" == "--keep" ]]; then
    KEEP_COMPRESSED=true
  else
    DIRS+=("$arg")
  fi
done

# Check if at least one directory is provided
if [[ ${#DIRS[@]} -lt 1 ]]; then
  echo "Usage: sbatch $0 dir1 [dir2 dir3 ...] [--keep]" >&2
  exit 1
fi

THREADS=${SLURM_CPUS_PER_TASK:-1}   # Use all cores Slurm gives us

# ---- Environment for multithreaded zstd ------------------------------------
export ZSTD_NBTHREADS="$THREADS"    # Honor -T0 (auto) inside zstd

# Set remove flag based on user preference
if $KEEP_COMPRESSED; then
  REMOVE_FLAG=""
  echo "Decompressing files in '${DIRS[*]}' using ${THREADS} threads. Keeping original compressed files."
else
  REMOVE_FLAG="--rm"
  echo "Decompressing files in '${DIRS[*]}' using ${THREADS} threads. Removing original compressed files."
fi

# ---- Main decompression loop -----------------------------------------------
# Process each directory
for DIR in "${DIRS[@]}"; do
  if [[ ! -d "$DIR" ]]; then
    echo "Warning: '$DIR' is not a directory, skipping." >&2
    continue
  fi

  echo "Processing directory: $DIR"
  
  # Using recursive decompression with zstd
  if $KEEP_COMPRESSED; then
    # Without --rm flag
    zstd -d -r -T0 --quiet "$DIR"
  else
    # With --rm flag to remove compressed files
    zstd -d -r -T0 --rm --quiet "$DIR"
  fi
  
  # Alternative implementation using find and xargs:
  # find "$DIR" -type f -name "*.zst" -print0 | 
  #   xargs -0 -n1 -P "$THREADS" zstd -d -T0 $REMOVE_FLAG --quiet
done

echo "Done."