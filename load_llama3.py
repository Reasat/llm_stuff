import os
os.environ['TRANSFORMERS_CACHE'] = '/home/phison/Desktop/llm'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#model_id = "/home/phison/Desktop/llm/Llama-2-7b-hf"
#model_id = "/home/phison/Desktop/llm/Meta-Llama-3-8B-HF"
model_id = "meta-llama/Llama-2-70b-chat-hf"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",

)
tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        )

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

