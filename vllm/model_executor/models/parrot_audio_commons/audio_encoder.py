import torch

__all__ = ["_get_feat_extract_output_lengths"]

torch.set_float32_matmul_precision('high')

# === Audio Encoder === #

# From ParrotAudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Computes the output length of the parrot audio encoder
    """
    return input_lengths, input_lengths
