# Need to figure out 
# scoring function: BLEURT, BLEU, and ROUGE
# input scheme




# basic test 
python main.py model@_global_=gpt2-xl limit=15

# benchmark test
python main.py model@_global_=causal limit=15 tasks=truthfulqa_gen
python main.py model@_global_=causal limit=15 tasks=truthfulqa_mc2
python main.py model@_global_=causal limit=15 tasks=crows_pairs_english
python main.py model@_global_=causal limit=15 tasks=headqa_en
python main.py model@_global_=causal limit=15 tasks=ethics_cm
python main.py model@_global_=causal limit=15 tasks=ethics_deontology
python main.py -m model@_global_=causal limit=15 tasks=ethics_justice,ethics_utilitarianism_original,ethics_utilitarianism,ethics_virtue
python main.py model@_global_=causal limit=15 tasks=toxigen 
python main.py model@_global_=causal  limit=15 tasks=winogrande
python main.py model@_global_=causal  limit=15 tasks=realtoxicityprompts
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

python main.py model@_global_=chat  limit=15 model_name="PKU-Alignment/beaver-7b-v1.0" 
python main.py model@_global_=chat  limit=15 model_name="fblgit/juanako-7b-UNA" 
python main.py model@_global_=chat  limit=15 model_name="HuggingFaceH4/zephyr-7b-beta" 
python main.py model@_global_=chat  limit=15 model_name="PKU-Alignment/alpaca-7b-reproduced" 
python main.py model@_global_=chat  limit=15 model_name="openchat/openchat_3.5" 
python main.py model@_global_=chat  limit=15 model_name="mistralai/Mistral-7B-Instruct-v0.2" 
python main.py model@_global_=chat  model_name="OpenBuddy/openbuddy-mistral-7b-v13"  tasks=advbench

python main.py model@_global_=api  limit=15 model_name="gpt-4-0314"
# 70b model for later evaluation
python main.py model_name=shiningvaliant  limit=15 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-70b-hf" tasks=toxigen 

# test wandb
python main.py model_name=gpt2-xl limit=15 wandb.enabled=True


python main.py model@_global_=causal limit=15 tasks=truthfulqa_gen model_name=/ibex/ai/home/zhanw0g/zephyr-7b/sft-full-31651894

# Leaderboard
python main.py model@_global_=causal tasks=arc_challenge num_fewshot=25 model_name=/ibex/ai/home/zhanw0g/zephyr-7b/dpo-selective-32046771
python main.py model@_global_=causal tasks=hellaswag num_fewshot=10 model_name=/ibex/ai/home/zhanw0g/zephyr-7b/dpo-selective-32046771

python main.py model@_global_=chat  limit=15 tasks=alert

# alpaca
alpaca_eval evaluate_from_model --output_path /home/zhanw0g/github/lm-evaluation-harness/experiment/results/zephyr-pairrm-iter3 --model_configs /home/zhanw0g/github/lm-evaluation-harness/experiment/helpful_config/zephyr-pairrm-iter3 --max_instances 10


# Decodingtrust
dt-run +model_config=zephyr_selective +advglue=alpaca ++advglue.resume=True
dt-run +model_config=zephyr_selective +advglue=benign 