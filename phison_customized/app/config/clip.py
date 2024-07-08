from utils.utils import load_hf_tokenizer, load_hf_featureextract
from ..dataloader.clip import prepare_clip_dataloader
from transformers import CLIPImageProcessor
from ..eval import *

config_dict = {
    'tokenizer': load_hf_tokenizer,
    'prepare_function': prepare_clip_dataloader, 
    'feature_extractor_function': load_hf_featureextract,
    'eval_type': LanguageModelEval
}