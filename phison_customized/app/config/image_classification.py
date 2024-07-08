from utils.utils import load_hf_tokenizer, load_hf_featureextract
from ..dataloader.image_classification import prepare_image_classification_dataloader
from transformers import CLIPImageProcessor
from ..eval import *

config_dict = {
    'prepare_function': prepare_image_classification_dataloader, 
    'eval_type': VisionModelEval,
}