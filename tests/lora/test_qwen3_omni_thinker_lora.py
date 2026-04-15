# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)


def _write_adapter(
    adapter_dir: Path,
    *,
    target_modules: list[str],
    tensors: dict[str, torch.Tensor],
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
    }
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))


def test_qwen3_omni_thinker_lora_mapper_and_skip_prefixes(tmp_path: Path):
    adapter_dir = tmp_path / "qwen3_omni_lora"
    tensors = {
        (
            "base_model.model.thinker.model.layers.0.self_attn.q_proj"
            ".lora_A.weight"
        ): torch.randn(8, 16, dtype=torch.float16),
        (
            "base_model.model.thinker.model.layers.0.self_attn.q_proj"
            ".lora_B.weight"
        ): torch.randn(16, 8, dtype=torch.float16),
        (
            "base_model.model.thinker.visual.blocks.0.attn.qkv"
            ".lora_A.weight"
        ): torch.randn(8, 16, dtype=torch.float16),
        (
            "base_model.model.thinker.visual.blocks.0.attn.qkv"
            ".lora_B.weight"
        ): torch.randn(16, 8, dtype=torch.float16),
    }
    _write_adapter(
        adapter_dir,
        target_modules=["q_proj", "qkv"],
        tensors=tensors,
    )

    peft_helper = PEFTHelper.from_local_dir(adapter_dir, max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(
        str(adapter_dir),
        expected_lora_modules={"q_proj"},
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        weights_mapper=Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper,
        skip_prefixes=Qwen3OmniMoeThinkerForConditionalGeneration.lora_skip_prefixes,
    )

    loaded_modules = set(lora_model.loras)
    assert loaded_modules == {"language_model.model.layers.0.self_attn.q_proj"}


def test_qwen3_omni_thinker_lora_per_expert_module_load(tmp_path: Path):
    adapter_dir = tmp_path / "qwen3_omni_expert_lora"
    tensors = {
        (
            "base_model.model.thinker.model.layers.0.mlp.experts.0.up_proj"
            ".lora_A.weight"
        ): torch.randn(8, 16, dtype=torch.float16),
        (
            "base_model.model.thinker.model.layers.0.mlp.experts.0.up_proj"
            ".lora_B.weight"
        ): torch.randn(16, 8, dtype=torch.float16),
    }
    _write_adapter(
        adapter_dir,
        target_modules=["experts.0.up_proj"],
        tensors=tensors,
    )

    peft_helper = PEFTHelper.from_local_dir(adapter_dir, max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(
        str(adapter_dir),
        expected_lora_modules={"experts.0.up_proj"},
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        weights_mapper=Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper,
    )

    loaded_modules = set(lora_model.loras)
    assert loaded_modules == {"language_model.model.layers.0.mlp.experts.0.up_proj"}


def test_qwen3_omni_thinker_lora_rejects_unexpected_module(tmp_path: Path):
    adapter_dir = tmp_path / "qwen3_omni_bad_lora"
    tensors = {
        (
            "base_model.model.thinker.model.layers.0.self_attn.q_proj"
            ".lora_A.weight"
        ): torch.randn(8, 16, dtype=torch.float16),
        (
            "base_model.model.thinker.model.layers.0.self_attn.q_proj"
            ".lora_B.weight"
        ): torch.randn(16, 8, dtype=torch.float16),
    }
    _write_adapter(
        adapter_dir,
        target_modules=["q_proj"],
        tensors=tensors,
    )

    peft_helper = PEFTHelper.from_local_dir(adapter_dir, max_position_embeddings=4096)
    with pytest.raises(ValueError, match="Please verify that the loaded LoRA module is correct"):
        LoRAModel.from_local_checkpoint(
            str(adapter_dir),
            expected_lora_modules={"k_proj"},
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            weights_mapper=Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper,
        )
