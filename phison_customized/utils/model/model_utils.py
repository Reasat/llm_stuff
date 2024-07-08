# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
 
# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
)
import transformers
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig
import torch.nn.functional as F

def mlp_forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


## Ref. https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
    topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    topk_weight = topk_weight.to(hidden_states.dtype)

    hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
    y = torch.empty_like(hidden_states)
    flat_topk_idx = topk_idx.view(-1)
    for i in range(self.num_experts):
        expert = self.experts[i]
        y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def replace_mixtral_moe_with_dense_impl():
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP

    MixtralBLockSparseTop2MLP.forward = mlp_forward
    MixtralSparseMoeBlock.forward = moe_forward
 
def print_rank_0(message, verbose):
    if verbose:
        print(message)

def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""
    flash_attn_support = True
    for device_id in range(torch.cuda.device_count()):
        try:
            # List of allowed architectures for AMD GPU
            allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"] 
            if torch.cuda.get_device_properties(device_id).gcnArchName not in allowed_archs:
                flash_attn_support = False
        except:
            major, _ = torch.cuda.get_device_capability(device_id)
            if major<8: # allowed architecture versions for Nvidia GPU
                flash_attn_support = False
    return flash_attn_support
 
def create_hf_model(model_name_or_path = None,
                    tokenizer = None,
                    ds_config = None,
                    use_flash_attention_2 = True,
                    gradient_checkpointing = True,
                    verbose = True):
 
    assert model_name_or_path is not None, "require model_name_or_path to load the model or config"
   
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    MODEL_NAME = model_config.architectures[0]
    MODEL_CLASS = getattr(transformers, MODEL_NAME)
 
   
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        print_rank_0(f"[INFO] Partition the model with config file setting",verbose=verbose)
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        print_rank_0(f"[INFO] The model is not partitioned. Please make sure the VRAM is enough during training",verbose=verbose)
        dschf = None
 
    if use_flash_attention_2 and supports_flash_attention():
        try:
            model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            use_flash_attention_2=use_flash_attention_2,
                                            config=model_config)
            
            replace_mixtral_moe_with_dense_impl()
            print(f'[INFO] Replace MixtralBLockSparseTop2MLP.forward and MixtralSparseMoeBlock.forward with dense implementation')
            print_rank_0(f"[INFO] You are loading [{MODEL_NAME}] with FlashAttention2",verbose=verbose)
        except:
            model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=model_config)
            print_rank_0(f"[INFO] [{MODEL_NAME}] does not support FlashAttention2",verbose=verbose)
    elif use_flash_attention_2==True and supports_flash_attention()==False:
        model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=model_config)
        print_rank_0(f"[INFO] Current GPUs do not support FlashAttention2",verbose=verbose)
    else:
        model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=model_config)
   
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print_rank_0(f"[INFO] You are setting [{MODEL_NAME}] with Gradient Checkpointing",verbose=verbose)
        except:
            print_rank_0(f"[INFO] [{MODEL_NAME}] does not support Gradient Checkpointing",verbose=verbose)
   
    if "ForCausalLM" in MODEL_NAME:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
    return model
 