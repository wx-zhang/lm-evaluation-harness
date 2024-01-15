#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J alpaca
#SBATCH -o ./logs/llm/%J.out
#SBATCH -e ./logs/llm/%J.err
#SBATCH --account=conf-eccv-2024.03.14-elhosemh
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G

source /home/zhanw0g/anaconda3/bin/activate
conda activate alignment



# Alpaca
alpaca_eval evaluate_from_model --output_path /home/zhanw0g/github/alpaca_eval/results/$1 --model_configs /home/zhanw0g/github/alpaca_eval/src/alpaca_eval/models_configs/$1