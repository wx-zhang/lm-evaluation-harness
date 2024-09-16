#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J alpaca
#SBATCH -o ./logs/alpaca/%J.out
#SBATCH -e ./logs/alpaca/%J.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G

source /home/zhanw0g/anaconda3/bin/activate
conda activate alignment



# Alpaca
alpaca_eval evaluate_from_model --output_path /home/zhanw0g/github/lm-evaluation-harness/experiment/results/$1 --model_configs /home/zhanw0g/github/lm-evaluation-harness/experiment/helpful_config/$1 