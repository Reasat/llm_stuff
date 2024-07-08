import os
import shutil
import argparse
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from utils.utils import save_hf_format
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from utils.data.data_utils import create_prompt_dataset
from torch.utils.data import DataLoader, RandomSampler
import deepspeed
from deepspeed import comm as dist
from transformers.deepspeed import HfDeepSpeedConfig
import time
from datetime import datetime

def function_filter(x):
    x = x.split('#')[0]
    if 'epoch' in x and 'step' in x:
        if len(x.split('_')) == 5:
            return True
        else:
            return False
    else:
        return False

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def prepare_dataloader(tokenizer,
                       dataset_name,
                       batch_size,
                       target_max_length,
                       seed,
                       train_phase: int = 1):
    
    train_dataset, eval_dataset = create_prompt_dataset(
        dist.get_rank(),
        dataset_name,
        '10,0,0',
        '/tmp/data_files/',
        train_phase,
        seed,
        tokenizer,
        target_max_length,
        end_of_conversation_token=tokenizer.eos_token,
        reload=True)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  num_workers=4,
                                  prefetch_factor=4)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=1)

    return train_dataloader, eval_dataloader

def print_rank_0(msg, rank):
    if rank == 0:
        print(msg)

def training_function(config, args):
    # Initialize accelerator
    base_model = args.model_name_or_path
    data_path = args.data_path
    gradient_accumulation_steps = args.gradient_accumulation_steps
    __per_device_train_batch_size = args.per_device_train_batch_size
    __max_seq_len = args.max_seq_len
    __seed = config["seed"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_config = AutoConfig.from_pretrained(base_model)
    ds_config = __get_train_ds_configv3()
    dschf = HfDeepSpeedConfig(ds_config)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision="bf16")
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    
    set_seed(seed)
    train_dataloader, eval_dataloader = prepare_dataloader(tokenizer, 
                                                           data_path, 
                                                           __per_device_train_batch_size, 
                                                           __max_seq_len, 
                                                           __seed)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            from_tf=bool(".ckpt" in base_model),
            use_flash_attention_2=True,
            config=model_config
        )
    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    __train_batch_size = __per_device_train_batch_size * torch.distributed.get_world_size() *  gradient_accumulation_steps


    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    timer_start = time.time()
    task_name = f'{datetime.now():%Y-%m-%d-%H-%M-%S}'
    if args.model_name_or_path[-1] == '/':
        args.model_name_or_path = args.model_name_or_path[:len(args.model_name_or_path)-1]
    if('#' in args.model_name_or_path):
        model_basename = args.model_name_or_path.split("#")[-1]
    else:
        model_basename = (args.model_name_or_path).split("/")[-1]

    
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step == args.max_iter:
                break

            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = to_device(batch, accelerator.device)
            print_rank_0(f"[{datetime.now()}] [PHISON START] Epoch: {epoch}, Iteration: {step}", dist.get_rank)
            print_rank_0(f"[{datetime.now()}] [Forward][start]", dist.get_rank())
            t1 = time.time()
            outputs = model(**batch)
            t2 = time.time()
            print_rank_0(f"[{datetime.now()}] [Forward][time spent]:{t2-t1}", dist.get_rank())
            loss = outputs.loss
            loss_item = loss.item()
            print_rank_0(f"[{datetime.now()}] [Loss]:{loss_item}", dist.get_rank()) 
            loss = loss / gradient_accumulation_steps            

            # Backward
            print_rank_0(f"[{datetime.now()}] [Backward][Start]", dist.get_rank())
            t1 = time.time()
            accelerator.backward(loss)
            t2 = time.time()
            print_rank_0(f"[{datetime.now()}] [Backward][time spent]:{t2-t1}", dist.get_rank())
            # Optimizer Updata
            if (step+1) % gradient_accumulation_steps == 0:
                print_rank_0(f"[{datetime.now()}] [Update][Start]", dist.get_rank())
                t1 = time.time()
                optimizer.step()
                t2 = time.time()
                print_rank_0(f"[{datetime.now()}] [Update][time spent]:{t2-t1}", dist.get_rank())
                lr_scheduler.step()
                optimizer.zero_grad()
            if (step+1) % gradient_accumulation_steps==0:
                total_time = time.time()-timer_start
                print_rank_0(f"Training efficiency: {__train_batch_size*__max_seq_len/total_time} (tokens/s)\n", dist.get_rank())
                timer_start = time.time()
            print_rank_0(f"[{datetime.now()}] [PHISON END] Iteration: {step}\n", dist.get_rank())

            save_path = None
            if args.output_dir is not None:
                save_path = os.path.join(args.output_dir, f'finetuned_model_'+task_name)
            if save_path is not None and (step == len(train_dataloader)-1): 
                # gc.collect()
                dist.barrier()  
                
                save_step_path = os.path.join(save_path, f"epoch_{epoch}_step_{step}_#{model_basename}")
                os.makedirs(save_step_path, exist_ok=True)
                print_rank_0(f'saving the final model at epoch_{epoch}_step_{step} ...', dist.get_rank)

                if dist.get_rank == 0:
                    save_hf_format(model, save_step_path, tokenizer)
                    model.save_16bit_model(save_step_path, save_filename='pytorch_model.bin')
                dist.barrier() 

                file_list = os.listdir(save_path)
                file_list = list(filter(function_filter, file_list))
                if len(file_list) > 5 and dist.get_rank() == 0:
                    sorted_list = sorted(file_list, key=lambda x:(int(x.split('_')[1]), int(x.split('_')[3])))
                    shutil.rmtree(os.path.join(save_path, sorted_list[0]))

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    
    ##################### Model arguments #####################
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    ################### Training arguments ####################
    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help='Path to the training dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-path dataset2-path ...')

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Initial learning rate (after the potential warmup period) to use.",)
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=7e-6,
        help="Initial learning rate (after the potential warmup period) to use.",)

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
        "--weight_decay",
        type=float,
        default=0.,
        help="Weight decay to use.")
    
    parser.add_argument(
         "--max_iter",
         type=int,
         default=-1,
         help="local_rank for distributed training on gpus")

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.")
    
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
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",)
    
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": args.learning_rate, "num_epochs": args.num_train_epochs, "seed": 42, "batch_size": args.per_device_train_batch_size}
    training_function(config, args)

def __get_train_ds_configv3():
    # Setting DeepSpeed ZeRO config
    zero_opt_dict = {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "offload_param": {
            "device": None,
            "nvme_path": None,
            "pin_memory": False,
            "buffer_count": 1,
            "buffer_size": 0,  
            "max_in_cpu": 0,

        },
        "offload_optimizer": {
            "device": None,
            "nvme_path": None,
            "pin_memory": False,
            "buffer_count": 4,
            "fast_init": True,
        },
        "load_from_fp32_weights": False,
        "stage3_param_persistence_threshold": 0,
        "stage3_max_live_parameters": 0,
        "stage3_prefetch_bucket_size": 0,
        "stage3_gather_16bit_weights_on_model_save": True,
        "memory_efficient_linear": True,
        "round_robin_gradients": True,
        "sub_group_size" : 0,
        "zero_hpz_partition_size_pmod": 1
    }

    # Setting DeepSpeed asynchronous-IO config
    ds_aio_dict = dict(
        block_size=4*1024*1024,
        queue_depth=32,
        thread_count=1,
        single_submit=False,
        overlap_events=True
    )

    # Setting DeepSpeed hybrid engine config
    hybrid_engine_dict = {
        "enabled": False,
        "max_out_tokens": 512,
        "inference_tp_size": 1,
        "release_inference_cache": False,
        "pin_parameters": False,
        "tp_gather_partition_size": 1,
    }

    # Setting activation checkpointing config
    act_check_dict = {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }

    # All together DeepSpeed config
    ds_config = {
        "train_batch_size":1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "aio": ds_aio_dict,
        "bf16":{"enabled": True,},
        "flops_profiler": {"enabled": False},
        # "gradient_clipping": 0.9,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": hybrid_engine_dict,
        "activation_checkpointing": act_check_dict,
        "model_info":{"model_name": "/media/user/model/Llama-2-7b-hf/"},
    }
    return ds_config

if __name__ == "__main__":
    main()