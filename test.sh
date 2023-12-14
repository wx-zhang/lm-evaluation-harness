# Need to figure out 
# scoring function: BLEURT, BLEU, and ROUGE
# input scheme




# basic test 
python main.py model@_global_=gpt2-xl limit=15

# benchmark test
python main.py model@_global_=causal limit=15 tasks=truthfulqa_gen
python main.py model@_global_=causal limit=15 tasks=truthfulqa_mc
python main.py model@_global_=causal limit=15 tasks=crows_pairs_english
python main.py model@_global_=causal limit=15 tasks=headqa_en
python main.py model@_global_=causal limit=15 tasks=ethics_cm
python main.py model@_global_=causal limit=15 tasks=ethics_deontology
python main.py -m model@_global_=causal limit=15 tasks=ethics_justice,ethics_utilitarianism_original,ethics_utilitarianism,ethics_virtue
python main.py model@_global_=causal limit=15 tasks=toxigen 
python main.py model@_global_=causal  limit=15 tasks=winogrande
python main.py model@_global_=causal  limit=15 tasks=advbench

# model test 
python main.py model@_global_=causal  limit=15 model_name="EleutherAI/gpt-j-6b"
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-7b-instruct" 
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-40b-instruct" 
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-7b" 
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-40b" 

python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-7b-chat-hf" 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-13b-chat-hf" 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-7b-hf" 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-13b-hf" 

python main.py model@_global_=causal  limit=15 model_name="PKU-Alignment/beaver-7b-v1.0" 
python main.py model@_global_=causal  limit=15 model_name="fblgit/juanako-7b-UNA" 
python main.py model@_global_=causal  limit=15 model_name="HuggingFaceH4/zephyr-7b-beta" 
python main.py model@_global_=causal  limit=15 model_name="PKU-Alignment/alpaca-7b-reproduced" 
python main.py model@_global_=causal  limit=15 model_name="openchat/openchat_3.5" 
python main.py model@_global_=causal  model_name="OpenBuddy/openbuddy-mistral-7b-v13"  tasks=ethics_cm

python main.py model@_global_=api  limit=15 model_name="gpt-4-0314"
# 70b model for later evaluation
python main.py model_name=shiningvaliant  limit=15 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-70b-hf" tasks=toxigen 

# test wandb
python main.py model_name=gpt2-xl limit=15 wandb.enabled=True


# Alpaca
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/openchat-v3.5-7b --output_path /ibex/project/c2106/wenxuan/alpaca/beaver-7b-v1.0 --max_instances 1
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/beaver-7b-v1.0  --max_instances 10
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/juanako-7b-UNA  --max_instances 1
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/zephyr-7b-beta  --max_instances 1
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/alpaca-7b-reproduced  --max_instances 1
alpaca_eval evaluate_from_model --model_configs /home/zhanw0g/github/lm-evaluation-harness/alpaca/mistral-7b-v13  --max_instances 1