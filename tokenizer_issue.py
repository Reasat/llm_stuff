import transformers 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os 
import torch 



pretrained = "facebook/opt-125m" # "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(pretrained,
                                              fast_tokenizer=True)
model_config = AutoConfig.from_pretrained(pretrained)
model = AutoModelForCausalLM.from_pretrained(
        pretrained,
        from_tf=bool(".ckpt" in pretrained),
        config=model_config)

print("original: ")
print(tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token)

output_dir = "./opt/"
model_to_save = model.module if hasattr(model, 'module') else model
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
save_dict = model_to_save.state_dict()
torch.save(save_dict, output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)

tokenizer_save = AutoTokenizer.from_pretrained(output_dir,
                                                fast_tokenizer=True)
print("after save: ")
print(tokenizer_save.pad_token, tokenizer_save.bos_token, tokenizer_save.eos_token)
