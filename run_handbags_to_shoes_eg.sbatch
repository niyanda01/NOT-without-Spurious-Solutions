#!/bin/bash
#SBATCH --job-name=not-handbags-eg-masked
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/users/kjlee01/NOT-without-Spurious-Solutions/logs/%x-%j.out
#SBATCH --error=/home/users/kjlee01/NOT-without-Spurious-Solutions/logs/%x-%j.err

source /opt/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate not-ot

cd /home/users/kjlee01/NOT-without-Spurious-Solutions

# Red handbags -> blue shoes, background removed (data/cache/ built by
# scripts/preprocess_product_data.py). Outputs: outputs/handbags_to_shoes/eg/
python -u scripts/run_handbags_to_shoes.py --method eg
