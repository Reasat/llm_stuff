#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n!pip install --no-deps xformers trl peft accelerate bitsandbytes\n')


# In[ ]:


get_ipython().system('pip install wandb -q')
output_dir = "/content/drive/MyDrive/Colab Notebooks/Llama 3"
import wandb
wandb.login()


# In[ ]:


from huggingface_hub import login
login()


# In[ ]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# In[ ]:


alpaca_prompt = """Below is an instruction in Bengali Language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("iamshnoo/alpaca-cleaned-bengali", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

# In[ ]:


from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size = 20,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 5,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        save_steps=50,
        save_total_limit=1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to="wandb",
    ),
)


# In[ ]:


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can change the instruction and input - leave the output blank!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "তিনি কত সালে গভর্নর নির্বাচিত হন", # instruction
        "শোয়ার্জনেগার হলিউড চলচ্চিত্রে অভিনয়ের মাধ্যমে পৃথিবীব্যাপী পরিচিতি লাভ করেন। দ্য টারমিনেটর, কোনান দ্য বার্বারিয়ান, প্রিডেটর তার অভিনীত উল্লেখযোগ্য চলচ্চিত্র। শোয়ার্জনেগার রিপাবলিকান পার্টির একজন পদপ্রার্থী হিসেবে ২০০৩ সালের অক্টোবরে ক্যালিফোর্নিয়ার গভর্নর নির্বাচিত হন এবং তৎকালীন গভর্নর গ্রে ডেভিসকে স্থলাভিষিক্ত করেন। ২০০৩ এর ২৩ নভেম্বর শোয়ার্জনেগার শপথ গ্রহণ করেন। পরবর্তীতে ২০০৬ সালের ৭ নভেম্বর তিনি ক্যালিফোর্নিয়ার গভর্নর হিসেবে পুনঃনির্বাচিত হন। এই নির্বাচনে তিনি ডেমোক্রেটিক পার্টি পদপ্রার্থী ফিল অ্যাঞ্জেলিডেসকে পরাজিত করেন।", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


from google.colab import userdata
token = userdata.get('HF_TOKEN')


# In[ ]:


import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save model and tokenizer to output_dir
model_path = os.path.join(output_dir, 'bangla-llama')
tokenizer_path = os.path.join(output_dir, 'bangla-llama')
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)


# In[ ]:


model.save_pretrained(model_path) # Local saving
tokenizer.save_pretrained(tokenizer_path)
model.push_to_hub("bangla-llama", token = token) # Online saving
tokenizer.push_to_hub("bangla-llama", token = token) # Online saving


# In[ ]:


import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save model and tokenizer to output_dir
model_path = os.path.join(output_dir, 'bangla-llama')
tokenizer_path = os.path.join(output_dir, 'bangla-llama')
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)


# In[ ]:


model.save_pretrained(model_path) # Local saving
tokenizer.save_pretrained(tokenizer_path)
model.push_to_hub("bangla-llama", token = token) # Online saving
tokenizer.push_to_hub("bangla-llamat", token = token) # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
alpaca_prompt.format(
        "তিনি কত সালে গভর্নর নির্বাচিত হন", # instruction
        "শোয়ার্জনেগার হলিউড চলচ্চিত্রে অভিনয়ের মাধ্যমে পৃথিবীব্যাপী পরিচিতি লাভ করেন। দ্য টারমিনেটর, কোনান দ্য বার্বারিয়ান, প্রিডেটর তার অভিনীত উল্লেখযোগ্য চলচ্চিত্র। শোয়ার্জনেগার রিপাবলিকান পার্টির একজন পদপ্রার্থী হিসেবে ২০০৩ সালের অক্টোবরে ক্যালিফোর্নিয়ার গভর্নর নির্বাচিত হন এবং তৎকালীন গভর্নর গ্রে ডেভিসকে স্থলাভিষিক্ত করেন। ২০০৩ এর ২৩ নভেম্বর শোয়ার্জনেগার শপথ গ্রহণ করেন। পরবর্তীতে ২০০৬ সালের ৭ নভেম্বর তিনি ক্যালিফোর্নিয়ার গভর্নর হিসেবে পুনঃনির্বাচিত হন। এই নির্বাচনে তিনি ডেমোক্রেটিক পার্টি পদপ্রার্থী ফিল অ্যাঞ্জেলিডেসকে পরাজিত করেন।", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
tokenizer.batch_decode(outputs)


# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# In[ ]:


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("bangla-llama-16bit", tokenizer, save_method = "merged_16bit", token = token)


# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if True: model.push_to_hub_merged("bangla-llama-lora", tokenizer, save_method = "lora", token = token)

# Merge to 4bit #This should be the last(After saving all models, guff, 16bits and all) one to save, make it True to do that. 
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if True: model.push_to_hub_merged("bangla-llama-4bit", tokenizer, save_method = "merged_4bit_forced", token = token)


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# In[ ]:


# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("bangla-llama-3", tokenizer,)
# if True: model.push_to_hub_gguf("bangla-llama-instruct_gguf_8bit", tokenizer, token = token)

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("bangla-llama-3", tokenizer, quantization_method = "f16")
if True: model.push_to_hub_gguf("bangla-llama-instruct_gguf_16bit", tokenizer, quantization_method = "f16", token = token)

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("bangla-llama-3", tokenizer, quantization_method = "q4_k_m")
if True: model.push_to_hub_gguf("bangla-llama-instruct_gguf_q4_k_m", tokenizer, quantization_method = "q4_k_m", token = token)


# In[ ]:


runtime.unassign()

