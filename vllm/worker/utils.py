# SPDX-License-Identifier: Apache-2.0
'''
Worker-related helper functions.
'''

import torch
from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase


def assert_enc_dec_mr_supported_scenario(
        enc_dec_mr: GPUModelRunnerBase) -> None:
    '''
    Asserted that the provided encoder/decoder model runner instance reflects
    a supported scenario.
    '''

    # Reminder: Please update docs/features/compatibility_matrix.md
    # If the feature combo become valid

    if enc_dec_mr.cache_config.enable_prefix_caching:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE'])

    if enc_dec_mr.sliding_window is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SWA'])

    if enc_dec_mr.scheduler_config.chunked_prefill_enabled:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL'])

    if getattr(enc_dec_mr.model_config.hf_config, 'attn_logit_softcapping',
               None) is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP']
        )

    if enc_dec_mr.lora_config is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LORA'])

    if enc_dec_mr.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PP'])

    if enc_dec_mr.scheduler_config.num_lookahead_slots > 0:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SPEC_DEC'])

    if enc_dec_mr.prompt_adapter_config is not None:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER'])


def get_model_dtype_v1(model_config) -> torch.dtype:
    if model_config.hf_config.model_type in ("parrot_audio", "parrot2_audio", "parrot2_audio_moe"):
        return model_config.hf_config.audio_config.torch_dtype
    return model_config.dtype


def get_model_dtype_v2(model_config) -> torch.dtype:
    default_dtype = model_config.dtype
    hf_config = model_config.hf_config
    if hasattr(hf_config, "audio_config"):
        if hasattr(hf_config.audio_config, "torch_dtype"):
            dtype = hf_config.audio_config.torch_dtype
        else:
            dtype = default_dtype
    elif hasattr(hf_config, "vision_config"):
        if hasattr(hf_config.vision_config, "torch_dtype"):
            dtype = hf_config.vision_config.torch_dtype
        else:
            dtype = default_dtype
    else:
        dtype = default_dtype
    return dtype
