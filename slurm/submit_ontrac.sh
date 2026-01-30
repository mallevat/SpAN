#!/bin/bash
#SBATCH --job-name=ontrac
#SBATCH --account=mbuchakj0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ontrac_%A_%a.out
#SBATCH --error=logs/ontrac_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# ONTraC SLURM Submission Script
# =============================================================================
# 
# Usage:
#   sbatch slurm/submit_ontrac.sh
#
# This runs ONTraC for all samples defined in SAMPLES array.
# Modify the paths and sample names below for your data.
# =============================================================================

# Activate conda environment
source ~/.bashrc
conda activate span_analysis

# Define samples
SAMPLES=(
    "MA-1"
    "MA-4"
    "MA-7"
    "MA-9"
    "MA-11"
    "MA-12"
)

# Get current sample from array index
SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

# Define paths (modify for your setup)
BASE_DIR="/nfs/turbo/umms-mbuchakj1/MikeAllevato"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/ontrac_analysis/per_sample_ontrac/${SAMPLE}"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=========================================="
echo "Running ONTraC for sample: ${SAMPLE}"
echo "Start time: $(date)"
echo "=========================================="

# Step 1: Prepare metadata (if not already done)
METADATA="${OUTPUT_DIR}/metadata.csv"
if [ ! -f "${METADATA}" ]; then
    echo "Preparing metadata..."
    python scripts/01_prepare_ontrac_metadata.py \
        --h5ad ${DATA_DIR}/${SAMPLE}.h5ad \
        --card_results ${DATA_DIR}/card_results/${SAMPLE}_proportions.csv \
        --output ${METADATA} \
        --sample_name ${SAMPLE}
fi

# Step 2: Run ONTraC
echo "Running ONTraC..."
python scripts/02_run_ontrac.py \
    --metadata ${METADATA} \
    --output_dir ${OUTPUT_DIR}/ontrac_results \
    --device cuda:0 \
    --epochs 1000 \
    --k_neighbors 50 \
    --hidden_feats 4 \
    --batch_size 5 \
    --seed 42

echo "=========================================="
echo "Completed sample: ${SAMPLE}"
echo "End time: $(date)"
echo "=========================================="
