# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn as nn

from vllm.config import LoadFormat, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)

logger = init_logger(__name__)


class MixedPrecisionModelLoader(DefaultModelLoader):
    """Mixed precision model loader that can load different file types from disk."""


    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with target_device:
            model = initialize_model(vllm_config=vllm_config,
                                        model_config=model_config)
        # Quantization does not happen in `load_weights` but after it
        self.load_weights(model, model_config)
        process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> tuple[str, list[str], bool]:
        # HINT: this is a hack to load the model in mixed precision
        self.load_config.load_format = LoadFormat.AUTO
        res = super()._prepare_weights(model_name_or_path, revision, fall_back_to_pt, allow_patterns_overrides)
        self.load_config.load_format = LoadFormat.MIXED_PRECISION
        return res