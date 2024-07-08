# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer, AutoFeatureExtractor
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import os
import subprocess
import psutil
import time
from filelock import FileLock
import collections

def print_rank_0(msg, rank=0):
    if rank <= 0:
        # print(msg)
        pass


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean

def load_hf_featureextract(args):
    model_name = args.model_name_or_path
    feature = AutoFeatureExtractor.from_pretrained(model_name)

    return feature

def load_hf_tokenizer(args, fast_tokenizer=False):
    model_name_or_path = args.model_name_or_path
    # if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
    # model_json = os.path.join(model_name_or_path, "config.json")
    # if os.path.exists(model_json):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                            fast_tokenizer=fast_tokenizer,
                                           )

    except:
        raise Exception(f'Tokenizer not found in {model_name_or_path}')
                # model_json_file = json.load(open(model_json))
                # model_name = model_json_file["_name_or_path"]
                # tokenizer = AutoTokenizer.from_pretrained(model_name,
                #                                         fast_tokenizer=fast_tokenizer,
                #                                         unk_token="<unk>",
                #                                         bos_token="<s>",
                #                                         eos_token="</s>")
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
    #                                               fast_tokenizer=fast_tokenizer)
    return tokenizer


def save_hf_format(model, output_dir, tokenizer=None, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    # WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    # output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # save_dict = model_to_save.state_dict()
    # for key in list(save_dict.keys()):
    #     if "lora" in key:
    #         del save_dict[key]
    # torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    
    if tokenizer:
        try:
            tokenizer.save_pretrained(output_dir)
        except:
            tokenizer.save_vocabulary(output_dir)


def _save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict 


def read_process_table(process_table_path):
    if os.path.exists(process_table_path):
        with open(process_table_path, 'r') as file:
            content = file.read()
            if content:  
                process_table = eval(content)
            else:
                process_table = {}  
    else:
        os.umask(0)
        fd = os.open(process_table_path, os.O_RDWR | os.O_CREAT, 0o777)
        process_table = {}

    return process_table


def write_process_table(process_table, process_table_path):
    with open(process_table_path, 'w') as file:
        file.write(str(process_table))



def check_process_status(process_table_path, process_table, nvme_path):
    print(process_table)
    # swap_paths = [os.path.join(nvme_path, swap_path) for swap_path in os.listdir(nvme_path) if swap_path.startswith('PhisonAI_') and os.path.join(nvme_path, swap_path) not in process_table.values()]


    # for swap_path in swap_paths:
    #     subprocess.run(['rm', '-r', swap_path])
    #     print(f'[INFO] remove swap data path: {swap_paths}')

    for pid in list(process_table.keys()):
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            swap_data_path = process_table[pid]['swap_data_path']
            subprocess.run(['rm', '-r', swap_data_path])
            print(f'[INFO] remove previous swap data path: {swap_data_path}')
            del process_table[pid]
    
    write_process_table(process_table, process_table_path)

def create_swap_data_path(nvme_path, local_rank):
    pid = os.getpid()
    parent_pid = os.getppid()
    swap_data_path = os.path.join(nvme_path, f'PhisonAI_{parent_pid}')
    if local_rank == 0:
        if os.path.exists(swap_data_path):
            subprocess.run(['rm', '-r', swap_data_path])

        os.makedirs(swap_data_path, exist_ok=True)
        os.chmod(swap_data_path, 0o777)
        process_table_path = os.path.join(nvme_path, 'process_table.txt')
        process_table = read_process_table(process_table_path)
        process_table[pid] = {'swap_data_path': swap_data_path}
        
        with FileLock(process_table_path):
            check_process_status(process_table_path, process_table, nvme_path)

    return swap_data_path


def print_debug(message, verbose=False):
    if verbose:
        print(message)

BUFFER_TEMP_TXT = "./.tmp/buffer.txt"

def create_buffer_temp_txt(global_rank, verbose=False):
    if global_rank == 0:
        if os.path.exists(BUFFER_TEMP_TXT):
            print_debug(f'[debug] remove {BUFFER_TEMP_TXT}', verbose)
            os.remove(BUFFER_TEMP_TXT)

        dir_path = os.path.dirname(BUFFER_TEMP_TXT)
        os.makedirs(dir_path, exist_ok=True)
        with open(BUFFER_TEMP_TXT, 'a') as file:
            print_debug(f'[debug] create {BUFFER_TEMP_TXT}', verbose)
        
        os.chmod(BUFFER_TEMP_TXT, 0o777)


def calculate_buffer(global_rank):
    MAX_REUSE_DISTANCE = 1e9
    
    print_rank_0("[Start] Calculate buffer specification", global_rank)
    file = open(BUFFER_TEMP_TXT, "r")
    buffer_spec = []
    final_buffer = {}
    line = file.readline()
    count = 0
    max_param = 0
    second_max_param = 0
    while line:
        param_id, param = line.split(" ")
        param = param.replace('\n', '')
        param = int(param)
        count += param
        if param > max_param:
            second_max_param = max_param
            max_param = param
        elif param > second_max_param and param != max_param:
            second_max_param = param
        if param not in final_buffer:
            final_buffer[param] = 1
        buffer_spec.append([param_id, param, count])
        line = file.readline()
    file.close() 
    
    temp_buffer = []
    for i in range(len(buffer_spec)):
        j = i+1
        record = [buffer_spec[i][0], buffer_spec[i][1], 0]
        if j>=len(buffer_spec):
            break
        while 1:
            if buffer_spec[j][0] == buffer_spec[i][0]:
                record[2] = buffer_spec[j][2] - buffer_spec[i][2]
            j+=1
            if j>=len(buffer_spec):
                break
            
            if buffer_spec[j][2] - buffer_spec[i][2] > MAX_REUSE_DISTANCE + second_max_param:
                break
        temp_buffer.append(record)
    temp_buffer.append(record)

    pre_buffer = []
    for i in range(len(temp_buffer)):
        if(temp_buffer[i][2]>0):
            param = temp_buffer[i][1]
            pre_buffer.append(param)
        else:
            pre_buffer = dict(collections.Counter(pre_buffer))
            for k, v in pre_buffer.items():
                if v > final_buffer[k]:
                    final_buffer[k] = v
            pre_buffer = []

    ASYNC_BLOCK_SIZE = 4*1024*1024 / 2 # BF16 format
    output_buffer = {}
    for k, v in final_buffer.items():
        if k >= ASYNC_BLOCK_SIZE:
            output_buffer[k] = v

    output_buffer = dict(collections.OrderedDict(sorted(output_buffer.items())))
    output_buffer = [(int(key), value) for key, value in output_buffer.items()]
    print_rank_0(f"Buffer specification: {output_buffer}", global_rank)

    return output_buffer


# Mutil user
def check_process_status(nvme_path):
    swap_file_names = os.listdir(nvme_path)
    swap_check_deict = {}

    for file_name in swap_file_names:
        if file_name.startswith('PhisonAI_') and '_' in file_name:
            key = int(file_name.split('_')[1])
            swap_check_deict[key] = os.path.join(nvme_path, file_name)


    for pid in list(swap_check_deict.keys()):
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            swap_data_path = swap_check_deict[pid]
            subprocess.run(['rm', '-r', swap_data_path])
            print(f'[INFO] remove previous swap data path: {swap_data_path}')

    

def create_swap_data_path(nvme_path, local_rank):
    parent_pid = os.getppid()
    swap_data_path = os.path.join(nvme_path, f'PhisonAI_{parent_pid}')
    if local_rank == 0:
        if os.path.exists(swap_data_path):
            subprocess.run(['rm', '-r', swap_data_path])

        os.makedirs(swap_data_path, exist_ok=True)
        os.chmod(swap_data_path, 0o777)
        check_process_status(nvme_path)

    return swap_data_path
