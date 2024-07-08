from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image



class ImageClassificationDataset(Dataset):
    def __init__(self, dataset, image_processor):
        self.image_processor = image_processor
        self.dataset_image = dataset["image"]
        self.dataset_label = dataset["label"]

    def __len__(self):
        return len(self.dataset_image)

    def __getitem__(self, idx):
        image = self.dataset_image[idx]
        if len(np.shape(image)) == 2:
            gray_three_channel = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            image = Image.fromarray(gray_three_channel)

        image = self.image_processor(image, return_tensors="pt", size={'shortest_edge':224})
        return {"pixel_values" : image["pixel_values"][0].to(torch.bfloat16), "labels" : self.dataset_label[idx]}

# resnet
def prepare_image_classification_dataloader(args):
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    
    train_dataset = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir="./dataset")
 
    train_dataset = ImageClassificationDataset(train_dataset, image_processor)  
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=4,
                                  prefetch_factor=4)
    

    
    return train_dataloader, None