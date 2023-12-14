from transformers import GPT2Tokenizer, GPT2Model
model = GPT2Model.from_pretrained('gpt2-xl')

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model.generate()

model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')
model.generate([1,2,3,4,5])