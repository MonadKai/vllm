# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_convert_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "convert_qwen3_omni_thinker_lora_keys.py"
    spec = importlib.util.spec_from_file_location(
        "convert_qwen3_omni_thinker_lora_keys", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_old_prefix_lora_format():
    module = _load_convert_script_module()

    fused_key = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
    per_expert_key = (
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
    )

    assert (
        module.detect_old_prefix_lora_format([fused_key])
        == module.LORA_FORMAT_FUSED_EXPERT
    )
    assert (
        module.detect_old_prefix_lora_format([per_expert_key])
        == module.LORA_FORMAT_PER_EXPERT
    )

    with pytest.raises(ValueError, match="Ambiguous old-prefix LoRA format"):
        module.detect_old_prefix_lora_format([fused_key, per_expert_key])

    with pytest.raises(ValueError, match="Unable to detect old-prefix LoRA format"):
        module.detect_old_prefix_lora_format(
            ["base_model.model.model.layers.0.mlp.up_proj.weight"]
        )


def test_convert_lora_key_format_fused_to_per_expert():
    module = _load_convert_script_module()

    fused_up = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
    fused_gate = "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight"

    fused_up_tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    fused_gate_tensor = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6)
    tensors = {
        fused_up: fused_up_tensor,
        fused_gate: fused_gate_tensor,
    }

    converted, changed_pairs = module.convert_lora_key_format(
        tensors,
        source_format=module.LORA_FORMAT_FUSED_EXPERT,
        target_format=module.LORA_FORMAT_PER_EXPERT,
        fused_to_per_expert_index=0,
    )

    expected_keys = {
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.1.up_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
        "base_model.model.model.layers.0.mlp.experts.1.gate_proj.lora_B.weight",
    }
    assert set(converted) == expected_keys
    assert len(changed_pairs) == 4

    torch.testing.assert_close(
        converted["base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"],
        fused_up_tensor[0],
    )
    torch.testing.assert_close(
        converted["base_model.model.model.layers.0.mlp.experts.1.up_proj.lora_A.weight"],
        fused_up_tensor[1],
    )


def test_convert_lora_key_format_fused_to_per_expert_requires_expert_axis():
    module = _load_convert_script_module()

    tensors = {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": torch.ones(3, 4),
    }
    with pytest.raises(
        ValueError,
        match="Expected expert dimension at axis 0",
    ):
        module.convert_lora_key_format(
            tensors,
            source_format=module.LORA_FORMAT_FUSED_EXPERT,
            target_format=module.LORA_FORMAT_PER_EXPERT,
            fused_to_per_expert_index=0,
        )


def test_convert_lora_key_format_fused_to_per_expert_requires_consistent_expert_count():
    module = _load_convert_script_module()

    tensors = {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": torch.ones(
            2, 3, 4
        ),
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.ones(
            3, 5, 6
        ),
    }
    with pytest.raises(
        ValueError,
        match="Inconsistent expert dimension across fused-expert tensors",
    ):
        module.convert_lora_key_format(
            tensors,
            source_format=module.LORA_FORMAT_FUSED_EXPERT,
            target_format=module.LORA_FORMAT_PER_EXPERT,
            fused_to_per_expert_index=0,
        )


def test_convert_lora_key_format_per_expert_to_fused_collision():
    module = _load_convert_script_module()

    per_expert_tensors = {
        (
            "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
        ): torch.ones(3, 4),
        "base_model.model.model.layers.0.mlp.experts.1.up_proj.lora_A.weight": (
            torch.zeros(3, 4)
        ),
    }

    with pytest.raises(ValueError, match="Key collision after LoRA format conversion"):
        module.convert_lora_key_format(
            per_expert_tensors,
            source_format=module.LORA_FORMAT_PER_EXPERT,
            target_format=module.LORA_FORMAT_FUSED_EXPERT,
            fused_to_per_expert_index=0,
        )


def test_convert_single_lora_key_format_noop_and_unsupported_conversion():
    module = _load_convert_script_module()
    key = "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"

    unchanged = module.convert_single_lora_key_format(
        key,
        source_format=module.LORA_FORMAT_PER_EXPERT,
        target_format=module.LORA_FORMAT_PER_EXPERT,
        fused_to_per_expert_index=0,
    )
    assert unchanged == key

    with pytest.raises(ValueError, match="Unsupported LoRA format conversion"):
        module.convert_single_lora_key_format(
            key,
            source_format="unknown_source_format",
            target_format=module.LORA_FORMAT_PER_EXPERT,
            fused_to_per_expert_index=0,
        )


def test_rename_keys_and_partition_lora_adapter_tensors():
    module = _load_convert_script_module()

    tensors = {
        (
            "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
        ): torch.ones(3, 4),
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.weight": torch.ones(
            3, 4
        ),
    }
    kept, dropped = module.partition_lora_adapter_tensors(tensors)
    assert set(kept) == {
        "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
    }
    assert dropped == ["base_model.model.model.layers.0.mlp.experts.0.up_proj.weight"]

    renamed, changed_pairs = module.rename_keys(
        kept,
        "base_model.model.model.",
        "base_model.model.language_model.model.",
    )
    assert set(renamed) == {
        "base_model.model.language_model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
    }
    assert changed_pairs == [
        (
            "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight",
            "base_model.model.language_model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight",
        )
    ]


def test_copy_adapter_config_for_output_clears_modules_to_save(tmp_path: Path):
    module = _load_convert_script_module()

    lora_dir = tmp_path / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    src_config_path = lora_dir / "adapter_config.json"
    src_config_path.write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "modules_to_save": ["lm_head"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_adapter_path = tmp_path / "out" / "adapter_model.safetensors"
    module.copy_adapter_config_for_output(
        lora_dir,
        output_adapter_path,
        dry_run=False,
    )

    out_config_path = output_adapter_path.parent / "adapter_config.json"
    assert out_config_path.is_file()
    out_config = json.loads(out_config_path.read_text(encoding="utf-8"))
    assert out_config["modules_to_save"] == []


def test_copy_adapter_config_for_output_dry_run_does_not_write(tmp_path: Path):
    module = _load_convert_script_module()

    lora_dir = tmp_path / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": ["lm_head"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    output_adapter_path = tmp_path / "out" / "adapter_model.safetensors"
    module.copy_adapter_config_for_output(
        lora_dir,
        output_adapter_path,
        dry_run=True,
    )

    assert not (output_adapter_path.parent / "adapter_config.json").exists()
