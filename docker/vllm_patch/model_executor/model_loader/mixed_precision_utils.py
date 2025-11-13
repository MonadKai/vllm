import logging

import torch
import torch.nn as nn
from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["ensure_model_precision", "cast_language_model_precision"]


def ensure_model_precision(
    model_config: ModelConfig,
    model: nn.Module,
) -> nn.Module:
    language_model_dtype = model_config.hf_config.text_config.torch_dtype
    audio_tower_dtype = model_config.hf_config.audio_config.torch_dtype
    multi_modal_projector_dtype = audio_tower_dtype
    if language_model_dtype != audio_tower_dtype:
        for name, param in model.audio_tower.named_parameters():
            if param.dtype != audio_tower_dtype:
                raise ValueError(f"audio_tower parameter {name} has dtype {param.dtype}, expected {audio_tower_dtype}!")
        for name, param in model.multi_modal_projector.named_parameters():
            if param.dtype != multi_modal_projector_dtype:
                raise ValueError(f"multi_modal_projector parameter {name} has dtype {param.dtype}, expected {multi_modal_projector_dtype}!")
        for name, param in model.language_model.named_parameters():
            if param.dtype != language_model_dtype:
                raise ValueError(f"language_model parameter {name} has dtype {param.dtype}, expected {language_model_dtype}!")


def cast_language_model_precision(
    model: nn.Module,
    language_model_dtype: torch.dtype,
) -> nn.Module:
    for name, param in model.language_model.named_parameters():
        if param.dtype != language_model_dtype:
            logger.warning(f"Casting language_model parameter {name} from {param.dtype} to {language_model_dtype}")
            param.data = param.data.to(language_model_dtype)
