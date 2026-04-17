#!/bin/bash
#SBATCH --account=project_462000963
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --job-name=LUMIO_copy_check
#SBATCH --output=../logs/lumio_transfer_%j.out
#SBATCH --error=../logs/lumio_transfer_%j.err

# ==============================================================================
# Configuration Variables
# ==============================================================================
# Local Source Directory to copy. This is the directory whose *contents* will be copied.
# Example: /scratch/project_462000000/my_input_data
SOURCE_DIR="$1"

# LUMI-O Destination Path (Bucket name and path prefix within the bucket)
# The local directory contents will be copied into this path.
# IMPORTANT: 'lumi-o' is the remote name configured by 'lumio-conf'.
DESTINATION_PATH="$2"

# ==============================================================================
# Data Transfer Execution
# ==============================================================================

echo "Starting LUMI-O data transfer and verification."
echo "Source: ${SOURCE_DIR}"
echo "Destination: ${DESTINATION_PATH}"
echo "-----------------------------------------"

# Load the lumio module (provides rclone and configuration)
module load lumio

# Check if rclone is available
if ! command -v rclone &> /dev/null
then
echo "Error: rclone command not found. Ensure 'module load lumio' succeeded."
exit 1
fi

# Perform the Copy Operation
# -P: show progress during transfer
# -v: verbose output
# --checkers 16: Use 16 parallel threads for hash/size checks (can speed up verification)
echo "Running rclone COPY operation..."
rclone copy -P -v "${SOURCE_DIR}" "${DESTINATION_PATH}"

COPY_EXIT_CODE=$?
if [ $COPY_EXIT_CODE -ne 0 ]; then
echo "ERROR: rclone copy failed with exit code $COPY_EXIT_CODE."
exit $COPY_EXIT_CODE
else
echo "SUCCESS: rclone copy completed."
fi

echo "-----------------------------------------"

# Perform the Verification Check
# Check for differences in size and hash. No data is transferred here.
echo "Running rclone CHECK operation for verification..."
rclone check -v "${SOURCE_DIR}" "${DESTINATION_PATH}"

CHECK_EXIT_CODE=$?
if [ $CHECK_EXIT_CODE -ne 0 ]; then
echo "WARNING: rclone check found differences or errors (Exit Code $CHECK_EXIT_CODE). See logs for details."
exit $CHECK_EXIT_CODE
else
echo "SUCCESS: rclone check passed. Source and destination are identical."
fi

echo "-----------------------------------------"
echo "Script finished successfully."