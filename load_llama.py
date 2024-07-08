import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#```python
# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
#model_path = 'brishtiteveja/bangla-llama-7b-instruct-v0.1'
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import load_checkpoint_in_model
import torch
from transformers import LlamaTokenizer
#model_path = "/home/phison/Desktop/llm/result/finetuned_model_2024-07-03-13-20-10/epoch_0_step_924_#Llama-2-70b-hf"
#model_path = "/home/phison/Desktop/llm/Meta-Llama-3-8B-Instruct"
model_path = "/home/phison/Desktop/llm/Llama-2-70b-hf"
with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(model_path)
# Number of GPUs available
model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map = 'auto',
        offload_folder = model_path,
        dtype = torch.float16,
        no_split_module_classes = ["LlamaDecoderLayer"]
        )

#tokenizer=LlamaTokenizer.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained(model_path)

prompt = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

#prompt = "Complete this test prompt by writing something in the Bengali script. Use Bengali unicode while writing your answer"

def generate_response(question, context):
    inputs = tokenizer([prompt.format(question, context, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_start = responses.find("### Response:") + len("### Response:")
    response = responses[response_start:].strip()
    return response

# Example Usage:

question = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর মৃত্যু কবে হয় ?"
context = "২০১৬ সালের ২৩ জুলাই হৃদরোগে আক্রান্ত হয়ে মহাশ্বেতা দেবী কলকাতার বেল ভিউ ক্লিনিকে ভর্তি হন। সেই বছরই ২৮ জুলাই একাধিক অঙ্গ বিকল হয়ে তাঁর মৃত্যু ঘটে। তিনি মধুমেহ, সেপ্টিসেমিয়া ও মূত্র সংক্রমণ রোগেও ভুগছিলেন।"
prompt_input = prompt.format(question, context, "")
inputs = tokenizer([prompt_input], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print('============================')
print(responses)
response_start = responses.find("### Response:") + len("### Response:")
response = responses[response_start:].strip()
print('============================')
print(response)
