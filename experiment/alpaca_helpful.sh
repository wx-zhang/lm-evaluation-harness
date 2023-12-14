#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J alpaca
#SBATCH -o /home/zhanw0g/logs/llm/%J.out
#SBATCH -e /home/zhanw0g/logs/llm/%J.err
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200G






# Alpaca
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/openchat-v3.5-7b 
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/beaver-7b-v1.0  
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/juanako-7b-UNA 
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/zephyr-7b-beta  
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/alpaca-7b-reproduced  
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/mistral-7b-v13  