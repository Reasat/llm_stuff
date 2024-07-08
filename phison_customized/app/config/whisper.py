from utils.utils import load_hf_tokenizer, load_hf_featureextract
from ..dataloader.whisper import prepare_whisper_dataloader
from ..eval import *

config_dict = {
    'tokenizer': load_hf_tokenizer,
    'prepare_function': prepare_whisper_dataloader, 
    'pad': '[PAD]',
    'eval_type': LanguageModelEval
}