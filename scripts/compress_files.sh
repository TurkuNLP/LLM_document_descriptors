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
# Compress every file in a directory with zstd, skipping anything that already
# ends in .zst.  Removes originals only after a successful compression.
#
#   sbatch compress_files.sh  /path/to/dir  [compression-level]
#
# Arguments
#   $1  Directory to walk (required)
#   $2  Optional zstd level (1–22, default 19)
#
# Notes
#   • Each file is compressed **independently** so failures affect only
#     that file.
#   • Both intra-file threading (-T) and inter-file parallelism (xargs -P)
#     are used, taking advantage of all CPUs Slurm allocates.
###############################################################################

set -euo pipefail

# ---- Parse and validate user input -----------------------------------------
DIR=${1:-}
if [[ -z "$DIR" ]]; then
  echo "Usage: sbatch $0 /path/to/dir [compression-level]" >&2
  exit 1
fi

LEVEL=${2:-19}                      # Default to a high compression level
THREADS=${SLURM_CPUS_PER_TASK:-1}   # Use all cores Slurm gives us

# ---- Environment for multithreaded zstd ------------------------------------
export ZSTD_NBTHREADS="$THREADS"    # Honour -T0 (auto) inside zstd

echo "Compressing files in '$DIR' with zstd -${LEVEL} using ${THREADS} threads."

# ---- Main compression loop --------------------------------------------------
# • find:   locate regular files that do **not** already end in .zst
# • xargs:  run zstd in parallel, one process per file, up to $THREADS at once
# • zstd:   -T0  = use all threads per process
#           --rm = delete the source file only if compression succeeds

zstd -r --rm -T0 --fast=4 --exclude-compressed --quiet "$DIR"

#find "$DIR" -type f ! -name '*.zst' -print0 |
#  xargs -0 -n1 -P "$THREADS" zstd -$LEVEL -T0 --rm --quiet

echo "Done."
