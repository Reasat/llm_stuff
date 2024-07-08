# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 8


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


def get_train_ds_configv3(offload,
                          stage=3,
                          nvme_path: str = "",
                          enable_hybrid_engine=False,
                          inference_tp_size=1,
                          release_inference_cache=False,
                          pin_parameters=False,
                          tp_gather_partition_size=1,
                          max_out_tokens=512):

    device = "cpu" if offload else "none"
    if len(nvme_path) > 0:
        device = "nvme"
    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "offload_param": {
            "device": device,
            "nvme_path": nvme_path,
            "pin_memory": False,
            "buffer_count": 16,
            "buffer_size":1.6e9, # llama-7bhf:1.6e9,3e8
            "max_in_cpu": 0,

        },
        "offload_optimizer": {
            "device": device,
            "nvme_path": nvme_path,
            "pin_memory": False,
            "buffer_count": 8,
            "fast_init": False
        },
        "load_from_fp32_weights": False,
        "stage3_param_persistence_threshold": 0,
        "stage3_max_live_parameters": 0,
        "stage3_prefetch_bucket_size": 0,
        "memory_efficient_linear": True,
        "round_robin_gradients": False,
        # "zero_hpz_partition_size": 8,
        # "zero_quantized_weights": True,
        # "zero_quantized_gradients": True
    }
    ds_aio = dict(
        block_size=1048576*4,
        queue_depth=32,
        thread_count=1,
        single_submit=False,
        overlap_events=True
    )
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "aio": ds_aio,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100,
            "fp16_master_weights_and_gradients": True
        },
        # "amp": {
        #     "enabled": True,
        #     "opt_level ":"O2"
        # },
        "flops_profiler": {
            "enabled": False
        },
        "gradient_clipping": 0.9,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
          "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": True,
            "profile": True
            }
    }
