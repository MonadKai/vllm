import torch
import torch.nn as nn


__all__ = ["LinearAdaptor"]


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
