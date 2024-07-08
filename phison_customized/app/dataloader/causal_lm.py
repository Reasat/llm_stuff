
from utils.data.data_utils import create_prompt_dataset
from transformers import (
                          default_data_collator,
                          DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler




# ---------------- LLM ----------------
# This function is utilized to customize your training and validation dataset
def prepare_dataloader(args, train_phase: int = 1):
    tokenizer = args.tokenizer
    train_dataset, _ = create_prompt_dataset(args.local_rank,
                                                        args.data_path,
                                                        args.data_split,
                                                        args.data_output_path,
                                                        train_phase,
                                                        args.seed,
                                                        tokenizer,
                                                        args.max_seq_len,
                                                        end_of_conversation_token=tokenizer.eos_token,
                                                        reload=True) 
    
    train_sampler = DistributedSampler(train_dataset)

    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset,
                                        collate_fn=data_collator,
                                        sampler=train_sampler,
                                        batch_size=args.per_device_train_batch_size,
                                        pin_memory=True,
                                        prefetch_factor=4, 
                                        num_workers=4,
                                        persistent_workers=True)
    
    return train_dataloader, data_collator