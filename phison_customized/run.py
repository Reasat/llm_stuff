import argparse
import os
import math
import sys
import time
from torch.distributed import init_process_group
import logging
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed import comm as dist
from phisonMain import PhsionMain
import subprocess

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

os.environ["NCCL_IGNORE_DISABLED_P2P"]="1"
os.environ["NCCL_SHM_DISABLE"]="0"
os.environ["NCCL_SOCKET_IFNAME "]="eth"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--nvme_path', type=str, default="")
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=7e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9,
                        help="coefficients used for computing running averages of gradient. (first moment)")
    
    parser.add_argument("--beta2",
                        type=float,
                        default=0.95,
                        help="coefficients used for computing running averages of gradient. (second moment)")

    parser.add_argument("--eps",
                        type=float,
                        default=1e-8,
                        help="term added to the denominator to improve numerical stability.")

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    model = args.model_name_or_path.split(os.sep)[-1]
    swap_data_path = os.path.join(args.nvme_path, 'PhisonAI')
    if os.path.exists(swap_data_path) and args.local_rank == 0:
        subprocess.run(['rm', '-r', swap_data_path])

    log_folder = f"./phison/{model}"
    log_folder = os.path.join(log_folder, "nvme")
    log_folder = os.path.join(log_folder, f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
        
    init_process_group(backend="nccl")
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
    main_proc = PhsionMain.build_by_args(**param)
    
    
    logging.basicConfig(filename=os.path.join(log_folder, f'demo-r{dist.get_rank()}.log'), level=logging.INFO)
    logging.info(args.model_name_or_path)

    model, optimizer, train_dataloader, lr_scheduler = main_proc._get_init_model(args)
