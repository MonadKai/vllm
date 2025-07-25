import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig


__all__ = ["ParrotAudioMultiModalProjector"]


# === Multi Modal Projector === #

class LinearAdaptor(nn.Module):
    def __init__(self, encoder_dim: int, ffn_dim: int, llm_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.final_norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.final_norm(x)
        return x


# @support_torch_compile(dynamic_arg_dims={"audio_features": [0, 1]})
class ParrotAudioMultiModalProjector(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        text_config = config.text_config

        self.adaptor = LinearAdaptor(
            encoder_dim=audio_config.output_size,
            ffn_dim=config.adaptor_ffn_dim,
            llm_dim=text_config.hidden_size
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.adaptor(audio_features)
        return hidden_states
