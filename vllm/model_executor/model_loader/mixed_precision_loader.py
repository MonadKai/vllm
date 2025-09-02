import time
from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.config import LoadFormat, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from .loader import (
    DefaultModelLoader,
    _initialize_model,
    _process_weights_after_loading,
)

logger = init_logger(__name__)


class MixedPrecisionModelLoader(DefaultModelLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)

        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(
            self.get_all_weights(model_config, model)
        )
        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights
            - self.counter_before_loading_weights,
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

        _process_weights_after_loading(model, model_config, target_device)

        return model.eval()

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> Tuple[str, List[str], bool]:
        # HINT: this is a hack to load the model in mixed precision
        self.load_config.load_format = LoadFormat.AUTO
        res = super()._prepare_weights(
            model_name_or_path,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        )
        self.load_config.load_format = LoadFormat.MIXED_PRECISION
        return res
