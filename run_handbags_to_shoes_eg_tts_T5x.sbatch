#!/bin/bash
#SBATCH --job-name=not-handbags-eg_tts_T5x-masked
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/users/kjlee01/NOT-without-Spurious-Solutions/logs/%x-%j.out
#SBATCH --error=/home/users/kjlee01/NOT-without-Spurious-Solutions/logs/%x-%j.err

source /opt/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate not-ot

cd /home/users/kjlee01/NOT-without-Spurious-Solutions

# Two-timescale EG (Adam): lr_T=5e-4, lr_f=1e-4 (ratio 5), same schedule as
# run_handbags_to_shoes_eg.sbatch. Outputs: outputs/handbags_to_shoes/eg_tts_T5x/
python -u scripts/run_handbags_to_shoes.py --method eg --lr-T 5e-4 --lr-f 1e-4 \
    --out-dir outputs/handbags_to_shoes/eg_tts_T5x
