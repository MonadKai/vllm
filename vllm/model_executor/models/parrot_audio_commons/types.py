from typing import TypedDict
import torch


# # === Audio Inputs === #

class ParrotAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(num_audios, num_mel_bins, 3000)`"""

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`"""
