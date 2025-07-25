import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.compilation.decorators import support_torch_compile
from typing import Optional

from .sense_voice_small import SenseVoiceEncoderSmall

__all__ = ["ParrotAudioEncoder", "_get_feat_extract_output_lengths"]


torch.set_float32_matmul_precision('high')

# === Audio Encoder === #

# @support_torch_compile(dynamic_arg_dims={"input_features": [0, 1], "audio_feature_lengths": 0})
class ParrotAudioEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`ParrotAudioEncoderLayer`].

    Args:
        config: ParrotAudioEncoderConfig
    """

    # Ignore copy
    def __init__(self, 
        vllm_config: VllmConfig,
        prefix: str = ""
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        self.sense_voice_small = SenseVoiceEncoderSmall(
            input_size=audio_config.input_size,
            output_size=audio_config.output_size,
            attention_heads=audio_config.attention_heads,
            linear_units=audio_config.linear_units,
            num_blocks=audio_config.num_blocks,
            tp_blocks=audio_config.tp_blocks,
            dropout_rate=audio_config.dropout_rate,
            attention_dropout_rate=audio_config.attention_dropout_rate,
            normalize_before=audio_config.normalize_before,
            kernel_size=audio_config.kernel_size,
            sanm_shfit=audio_config.sanm_shfit
        )

    def forward(
        self,
        input_features: torch.Tensor,  # [16, 500, 560]
        attention_mask: Optional[torch.Tensor] = None,  # None
        audio_feature_lengths: Optional[torch.Tensor] = None,  # [16]
        head_mask: Optional[torch.Tensor] = None,  # None
        output_attentions: Optional[bool] = None,  # None
        output_hidden_states: Optional[bool] = None,  # None
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs_pad, olens = self.sense_voice_small(
            input_features,
            ilens = audio_feature_lengths
        )  # xs_pad: [16, 500, 512], olens: [16]
        return xs_pad, olens

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Computes the output length of the parrot audio encoder
        """
        return input_lengths, input_lengths


# From ParrotAudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Computes the output length of the parrot audio encoder
    """
    return input_lengths, input_lengths
