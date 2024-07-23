import os
os.environ['TRANSFORMERS_CACHE'] = '/home/phison/Desktop/llm'

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import load_checkpoint_in_model

#model_id = "/home/phison/Desktop/llm/Llama-2-7b-hf"
#model_id = "/home/phison/Desktop/llm/Llama-2-7b-chat-hf"
#model_id = "/home/phison/Desktop/llm/Llama-2-70b-chat-hf"
#model_id = "/home/phison/Desktop/llm/result/finetuned_model_2024-07-03-13-20-10/epoch_0_step_924_#Llama-2-70b-hf"
#model_id = "/home/phison/Desktop/llm/result/finetuned_model_2024-07-05-23-51-51/epoch_0_step_1848_#Llama-2-7b-hf"
#model_id = "/home/phison/Desktop/llm/result/finetuned_model_2024-07-07-04-15-31/epoch_0_step_924_#Llama-2-7b-chat-hf"
model_id = ""

#model_id = "/home/phison/Desktop/llm/Meta-Llama-3-8B-HF"
#model_id = "/home/phison/Desktop/llm/Meta-Llama-3-8B-Instruct"
#model_id = "/home/phison/Desktop/llm/Meta-Llama-3-70B-Instruct"


model_path = model_id

'''
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",

)
'''
with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(model_path)

model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map = 'auto',
        offload_folder = model_path,
        dtype = torch.float16,
        no_split_module_classes = ["LlamaDecoderLayer"]
        )

tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        )


'''
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
'''
prompt = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

question = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর মৃত্যু কবে হয় ?"
context = "২০১৬ সালের ২৩ জুলাই হৃদরোগে আক্রান্ত হয়ে মহাশ্বেতা দেবী কলকাতার বেল ভিউ ক্লিনিকে ভর্তি হন। সেই বছরই ২৮ জুলাই একাধিক অঙ্গ বিকল হয়ে তাঁর মৃত্যু ঘটে। তিনি মধুমেহ, সেপ্টিসেমিয়া ও মূত্র সংক্রমণ রোগেও ভুগছিলেন।"
messages = prompt.format(question, context, "")
input_ids = tokenizer(messages, return_tensors="pt").to(model.device)['input_ids']
terminators = None

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0]
print('============ INPUT ===================')
print(tokenizer.decode(*input_ids))
print('============ OUTPUT ===================')
print(tokenizer.decode(response))

