#```python
# Use a pipeline as a high-level helper
from transformers import pipeline
#pipe = pipeline("text-generation", model="asif00/bangla-llama", 
        #ignore_mismatched_sizes = True
#        )
#```

#```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = 'brishtiteveja/bangla-llama-7b-instruct-v0.1'
#model_path = 'asif00/bangla-llama'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
if ~next(model.parameters()).is_cuda:
    model = model.cuda()
#```

# General Prompt Structure: 

#```python
prompt = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""
#```

# To get a cleaned up version of the response, you can use the `generate_response` function:

#```python
def generate_response(question, context):
    inputs = tokenizer([prompt.format(question, context, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_start = responses.find("### Response:") + len("### Response:")
    response = responses[response_start:].strip()
    return response
#```

# Example Usage:

#```python
question = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর মৃত্যু কবে হয় ?"
context = "২০১৬ সালের ২৩ জুলাই হৃদরোগে আক্রান্ত হয়ে মহাশ্বেতা দেবী কলকাতার বেল ভিউ ক্লিনিকে ভর্তি হন। সেই বছরই ২৮ জুলাই একাধিক অঙ্গ বিকল হয়ে তাঁর মৃত্যু ঘটে। তিনি মধুমেহ, সেপ্টিসেমিয়া ও মূত্র সংক্রমণ রোগেও ভুগছিলেন।"
answer = generate_response(question, context)
print(answer)
#```

