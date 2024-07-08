from utils.utils import load_hf_tokenizer
from ..dataloader.causal_lm import prepare_dataloader
from ..eval import *
from datasets import load_dataset

def return_prompt_and_responses(samples):
    return {
    "prompt": [
    f"### Input: ```{input}```\n ### Output: "
    for input in samples["question"]
    ],
    "chosen": samples["chosen"],
    "rejected": samples["rejected"],
    }




dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
original_columns = dataset.column_names
dataset = dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
    )

config_dict = {
    'tokenizer': load_hf_tokenizer,
    'prepare_function': prepare_dataloader, 
    'pad': '[PAD]',
    'eval_type': LanguageModelEval,
    'dataset': dataset,
}



