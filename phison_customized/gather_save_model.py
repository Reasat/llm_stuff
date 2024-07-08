import torch
import pickle
import os
from collections import OrderedDict
import time
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gather and save pickle files to pytorch_model.bin file.")
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing pickle files')
    args = parser.parse_args()
    return args
    


def sort_file_names(file_name):
    return int(file_name.split('.')[0])

def gather_model_pickle_to_bin(folder):
    weights = OrderedDict()
    print(f'Processing folder: {folder} to gather pickle files')
    file_list = filter(lambda x:('pickle' in x), os.listdir(folder))
    file_list = sorted(file_list, key=sort_file_names)

    if len(file_list) == 0 or os.path.exists(os.path.join(folder, 'pytorch_model.bin')):
        print(f'No pickle files in {folder} or pytorch_model.bin already exists')
        return

    for file_name in file_list:
        start_time = time.time()
        file_path = os.path.join(folder, file_name)
                
        if file_name.endswith('.pickle'):
            with open(file_path, 'rb') as f:
                key = file_name.split('.pickle')[0].split('.')[1::]
                key = '.'.join(key)
                weights[key] = pickle.load(f)

    torch.save(weights, os.path.join(folder, 'pytorch_model.bin'))

    for file in file_list:
        file_path = os.path.join(folder, file)
        os.remove(file_path)

def gather_model_folder(folder_path):

    contents = os.listdir(folder_path)
    sub_folders = [os.path.join(folder_path, f) for f in contents if os.path.isdir(os.path.join(folder_path, f))] 
    # folder_path = '/home/user/Desktop/MK/deepspeedaddlog/llama_13b_chat/finetuned_model_2024-02-21-10-09-18/epoch_0_step_0_#Llama-2-13b-chat-hf'
    for folder in sub_folders:
        FLAG = False
        with open(os.path.join(folder, 'config.json'), 'r') as f:
            config = json.load(f)
            try:
                FLAG = config['save_checkpoint']
            except:
                print(f'No success_checkpoint in {folder}')

        if FLAG:
            gather_model_pickle_to_bin(folder)                

if __name__ == '__main__':
    args = parse_args()
    gather_model_folder(args.folder_path)
