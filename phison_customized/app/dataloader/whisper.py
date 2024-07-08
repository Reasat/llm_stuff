from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import List, Union, Dict, Any, Optional
import torch
from transformers import (WhisperFeatureExtractor,
                          WhisperTokenizer,
                          WhisperProcessor)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler




@dataclass
class DataCollatorSpeechSeq2SeqWithPadding():
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt").to(dtype=torch.bfloat16)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Whisper Dataset
def prepare_whisper_dataloader(args):
    def process_dataset(batch):  # Added self parameter
        # if 'whisper-large-v3' in args.model_name_or_path.lower():
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
        tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language="Hindi", task="transcribe")
        # elif 'whisper-large-v2' in args.model_name_or_path.lower():
        #     feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
        #     tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="Hindi", task="transcribe")

        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
        
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="Hindi", task="transcribe")
    common_voice = DatasetDict()
    test = load_dataset("mozilla-foundation/common_voice_11_0", 'hi')
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(process_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
    print("------ create dataset done ------")
    train_dataset = common_voice["train"]
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # DataLoaders creation:
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size,
                                    num_workers=4,
                                    prefetch_factor=4)

    return train_dataloader, data_collator