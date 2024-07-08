import torch
import gc
from tqdm import tqdm
from deepspeed.accelerator import get_accelerator
from utils.utils import get_all_reduce_mean

def print_rank_0(msg, rank):
    if rank == 0:
        print(msg)
        
class BaseModelEval():
    def __init__(self):
        """
        define your model evaluation metric baseline score
        """
        
    def to_device(self, batch, device): 
        """
        Please don't modify this function
        """
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    def __call__(self, args, model, eval_dataloader, device): 
        """ 
        define your model evaluation metric
        """
        raise NotImplementedError