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
python main.py model@_global_=causal  limit=15 tasks=advbench

# model test 
python main.py model@_global_=causal  limit=15 model_name="EleutherAI/gpt-j-6b"
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-7b-instruct" 
python main.py model@_global_=causal  limit=15 model_name="tiiuae/falcon-7b" 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-7b-chat-hf" 
python main.py model@_global_=causal  limit=15 model_name="meta-llama/Llama-2-7b-hf" 


python main.py model@_global_=causal  limit=15 model_name="PKU-Alignment/beaver-7b-v1.0" 
python main.py model@_global_=causal  limit=15 model_name="fblgit/juanako-7b-UNA" 
python main.py model@_global_=causal  limit=15 model_name="HuggingFaceH4/zephyr-7b-beta" 
python main.py model@_global_=causal  limit=15 model_name="PKU-Alignment/alpaca-7b-reproduced" 
python main.py model@_global_=causal  limit=15 model_name="openchat/openchat_3.5" 
python main.py model@_global_=causal  model_name="OpenBuddy/openbuddy-mistral-7b-v13"  tasks=ethics_cm
python main.py model@_global_=api  limit=15 model_name="gpt-4-0314"