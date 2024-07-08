import os
import sys
from deepspeed import comm as dist
import subprocess
from datetime import datetime
from torch.distributed import init_process_group
import logging
import torch
import deepspeed
import argparse
import importlib
from transformers import SchedulerType
# from phison_trainer import PhsionMain, print_rank_0
from phison_eval import PhsionMain, print_rank_0

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ["NCCL_IGNORE_DISABLED_P2P"]="1"
os.environ["NCCL_SHM_DISABLE"]="0"
os.environ["NCCL_SOCKET_IFNAME "]="eth"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    
    ################### Data Arguments ###################
    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help='Path to the training dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-path dataset2-path ...')
    
    parser.add_argument(
        '--data_split',
        default="10,0,0",
        help='Data splitting for training, validation, and testing set')
    
    parser.add_argument(
        '--nvme_path', 
        type=str, 
        default="Path to the NVMe directory.")
    
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.'
        'This needs to be on a local storage of a node (not on a shared storage)')
    
    ################### Models Arguments ###################
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default="causal_lm",
        help="Choose the config file name of the model, the config file must be located in the ./config folder.",
    )

    ################### Training Arguments ###################
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help='Assigning number of GPU.')
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",)
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",)
    
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",)
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=7e-6,
        help="Initial learning rate (after the potential warmup period) to use.",)
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.,
        help="Weight decay to use.")
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.")
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],)
    
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="coefficients used for computing running averages of gradient. (first moment)")
    
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="coefficients used for computing running averages of gradient. (second moment)")

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="term added to the denominator to improve numerical stability.")

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the model.")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="A seed for reproducible training.")
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus")
    
    parser.add_argument(
         "--max_iter",
         type=int,
         default=-1,
         help="local_rank for distributed training on gpus")

    parser.add_argument(
         "--pretrain",
         type=int,
         default=0,
         help="local_rank for distributed training on gpus")
    
    parser.add_argument(
         "--allingpu",
         type=int,
         default=0,
         help="local_rank for distributed training on gpus")
    
    parser.add_argument(
        "--show_record",
        type=str,
        default=None,
        help="show the records of resource usages and model loss during training .")
    
    ################### LoRA Arguments ###################
    parser.add_argument(
        '--lora',
        action='store_true')
    
    parser.add_argument(
        '--lora_weight',
        type=str,
        default=None,
        help="Path to pretrained Lora weight.")
    
    parser.add_argument(
         "--target_modules",
         type=list,
         default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_heed'],
         help="target modules to apply low rank adaption")
    
    parser.add_argument(
         "--task_type",
         type=str,
         default='CAUSAL_LM',
         help="training task type")

    parser.add_argument(
         "--rank",
         type=int,
         default=8,
         help="dimension of the low-rank matrices")
    
    parser.add_argument(
         "--lora_alpha",
         type=int,
         default=16,
         help="scaling factor for the weight matrices")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    param = {"model_name_or_path": args.model_name_or_path,
             "training_data_path": args.data_path,
             "data_output_path": args.data_output_path,
             "max_seq_len": args.max_seq_len,
             "per_device_train_batch_size": args.per_device_train_batch_size,
             "per_device_eval_batch_size": args.per_device_eval_batch_size,
             "gradient_accumulation_steps": args.gradient_accumulation_steps,
             "nvme_path": args.nvme_path,
             "local_rank": args.local_rank,
             "seed": args.seed
            }
    
    return args, param

if __name__ == "__main__":
    task_name = f'{datetime.now():%Y-%m-%d-%H-%M-%S}'
    args, param = parse_args()

    # Navigate to model path and NVMe path
    model = args.model_name_or_path.split(os.sep)[-1]
    
    swap_data_path = os.path.join(args.nvme_path, 'PhisonAI')
    if os.path.exists(swap_data_path) and args.local_rank == 0:
        subprocess.run(['rm', '-r', swap_data_path])
    #if os.path.exists("/tmp/data_files/") and args.local_rank == 0:
    #    subprocess.run(['rm', '-r', "/tmp/data_files"])

    # Checking logging directory
    log_folder = f"./phison/{model}"
    log_folder = os.path.join(log_folder, "nvme")
    log_folder = os.path.join(log_folder, f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    # Initialize DeepSpeed process
    init_process_group(backend="nccl")

    # Check input argument and initialize Phison Trainer
    trainer = PhsionMain.build_by_args(**param)
    
    global_rank = dist.get_rank()
    
    MODEL_RESOURCES_CONFIG = {
        'tokenizer': None,
        'prepare_function': None, 
        'feature_extractor_function': None,
        'pad': None,
        'eval_type': None,
    }

    CONFIG_LIBRARY_PATH = "app.config"
    config_module = importlib.import_module(f'{CONFIG_LIBRARY_PATH}.{args.config_name}')
    config_dict = config_module.config_dict

    for key in MODEL_RESOURCES_CONFIG.keys():
        MODEL_RESOURCES_CONFIG[key] = config_dict.get(key, None)

    print_rank_0(f"MODEL_RESOURCES_CONFIG: {MODEL_RESOURCES_CONFIG}", global_rank)

    args.tokenizer = tokenizer = None
    if MODEL_RESOURCES_CONFIG['tokenizer'] is not None:
        tokenizer = MODEL_RESOURCES_CONFIG['tokenizer'](args)
        if args.pretrain:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        
        if MODEL_RESOURCES_CONFIG['pad'] is not None:
            tokenizer.add_special_tokens({'pad_token': MODEL_RESOURCES_CONFIG['pad']})
            tokenizer.padding_side = 'right'
        args.tokenizer = tokenizer

    args.feature_extractor_function = None
    if MODEL_RESOURCES_CONFIG['feature_extractor_function'] is not None:
        feature_extractor_function = MODEL_RESOURCES_CONFIG['feature_extractor_function'](args)
        args.feature_extractor = feature_extractor_function

    if MODEL_RESOURCES_CONFIG['prepare_function'] is not None:
        eval_dataloader, data_collator = MODEL_RESOURCES_CONFIG['prepare_function'](args)
    else:
        raise ValueError(f'config file must contain prepare function, please check the config file')


    # Logging file setting
    logging.basicConfig(filename=os.path.join(log_folder, f'demo-r{dist.get_rank()}.log'), level=logging.INFO)
    logging.info(args.model_name_or_path)

    # Running Phison Trainer
    trainer.eval(args, task_name, eval_dataloader, tokenizer, MODEL_RESOURCES_CONFIG['eval_type'])