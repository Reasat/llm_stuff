from utils.utils import load_hf_tokenizer
from ..dataloader.causal_lm import prepare_dataloader
from ..eval import *

config_dict = {
    'tokenizer': load_hf_tokenizer,
    'prepare_function': prepare_dataloader, 
    'pad': '[PAD]',
    'eval_type': LanguageModelEval
}