#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J llm-eval
#SBATCH -o ./logs/openllm/%J.out
#SBATCH -e ./logs/openllm/%J.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G


#run the application

source /home/zhanw0g/anaconda3/bin/activate
conda activate llm
cd  /home/zhanw0g/github/lm-evaluation-harness
export HF_HOME=/ibex/user/zhanw0g/huggingface

$1