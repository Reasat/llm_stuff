# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re
import random
import os

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.

# you can define your own dataset here
class CustomJsonDataset():
    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_datasets = load_dataset('json', data_files=dataset_name)
        self.dataset_name = 'custom json dataset'
        self.dataset_name_clean = dataset_name.replace("/", "_")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset
    
    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt_and_chosen(self, sample):
        """
        modify this function to implement customized prompt format.
        """
        return "Human: " + sample['instruct'] + "Asistant: " + sample['output']


class PretrainDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        #super().__init__(output_path, seed, local_rank, dataset_name)
        self.output_path = output_path
        self.local_rank = local_rank
        file, extension = os.path.splitext(dataset_name)
        if 'json' in extension:
            self.raw_datasets = load_dataset('json', data_files=dataset_name)
        elif 'csv' in extension:
            self.raw_datasets = load_dataset('csv', data_files=dataset_name)
            
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name.replace("/", "_")
        print('succeed loading pretrain dataset')

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return random.sample(list(self.raw_datasets["train"]), 1)

    def get_prompt_and_chosen(self, sample):
        if sample['text'] is not None:
            return sample['text']
        return None

    
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if "json" in dataset_name:
            self.raw_datasets = load_dataset('json', data_files=dataset_name)
        else:
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    def get_prompt_and_chosen(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']


class AlpacaDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mhenrichsen/alpaca_2k"
        self.dataset_name_clean = "mhenrichsen_alpaca_2k"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return random.sample(list(self.raw_datasets["train"]), 1)


    def get_prompt_and_chosen(self, sample):
        return sample['text']

