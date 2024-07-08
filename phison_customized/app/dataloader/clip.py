from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (AutoFeatureExtractor,
                          AutoTokenizer)

def prepare_clip_dataloader(args):
    """
    Prepare the dataloader for training and evaluation clip 
    """
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
        )

    feature_extractor = args.feature_extractor
    def tokenize_captions(examples):
        """
        Process the data with tokenizer
        """
        captions = [caption for caption in examples[caption_column]]
        text_inputs = tokenizer(captions, max_length=77, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples
    
    def transform_images(examples):
        """
        Process the data with feature_extractor
        """
        image_size = 224
        transform = []
        transform.append(T.ToTensor())
        transform.append(T.Resize([image_size], interpolation=InterpolationMode.BICUBIC))
        transform.append(T.CenterCrop(image_size))
        transform.append(T.ConvertImageDtype(torch.float))
        transform.append(T.Normalize(feature_extractor.image_mean, feature_extractor.image_std))
        transform = T.Compose(transform) 
        examples["pixel_values"] = [transform(image) for image in examples[image_column]]
        return examples
    
    def collate_fn(examples):
        """
        Create data collector
        """
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
        }
    dataset = load_dataset(args.data_path[0])
    image_column = 'image'
    caption_column = 'text'
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        function=tokenize_captions,
        batched=True,
        num_proc=4,
        desc="Running tokenizer on train dataset",
    )
    train_dataset.set_transform(transform_images)

    

    # DataLoaders creation:
    train_sampler = DistributedSampler(train_dataset)
  
    train_dataloader = DataLoader(train_dataset,
                                    collate_fn=collate_fn,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size,
                                    num_workers=4,
                                    prefetch_factor=4)
    

    return train_dataloader, collate_fn