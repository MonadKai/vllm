#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batch rename LoRA tensor keys for Qwen3 Omni Thinker language_model loading.

Typical use case:
- LoRA was trained against Qwen3MoeForCausalLM and keys look like:
    base_model.model.model.layers.0....
- Qwen3OmniMoeThinkerForConditionalGeneration expects language model keys like:
    base_model.model.language_model.model.layers.0....

This script rewrites key prefixes in adapter weight files and copies
``adapter_config.json`` from ``--lora-dir`` next to the output adapter file.
Non-empty ``modules_to_save`` in the config is cleared (vLLM requires it empty).

Tensor keys that do not end with ``.lora_A.weight`` or ``.lora_B.weight`` are treated
as invalid for this adapter and are omitted from the output after a warning.

Supported files:
- adapter_model.safetensors (default)
- adapter_model.bin
- adapter_model.pt

Examples:
    # Preview changes only
    python scripts/rename_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora --dry-run

    # Default: write beside input as <adapter>.renamed.<ext>
    python scripts/rename_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora

    # Write to a directory (default filename: adapter_model.safetensors)
    python scripts/rename_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora \
      --output-dir /path/to/converted_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

# Default output weights filename when using --output-dir (always safetensors).
DEFAULT_OUTPUT_ADAPTER_FILENAME = "adapter_model.safetensors"

# Standard PEFT LoRA weight suffixes; other tensors are warned and skipped.
LORA_ADAPTER_KEY_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def detect_adapter_file(lora_dir: Path) -> Path:
    candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    )
    for filename in candidates:
        path = lora_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No adapter weight file found in {lora_dir}. "
        f"Expected one of: {', '.join(candidates)}"
    )


def load_tensors(input_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    suffix = input_path.suffix
    if suffix == ".safetensors":
        metadata: dict[str, str] | None = None
        with safe_open(str(input_path), framework="pt") as reader:
            metadata = reader.metadata()
        tensors = safetensors_load_file(str(input_path))
        return tensors, metadata

    if suffix in {".bin", ".pt"}:
        obj = torch.load(str(input_path), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise ValueError(f"Unsupported checkpoint content type: {type(obj)}")
        # Filter only tensor entries and keep exact key names.
        tensors: dict[str, torch.Tensor] = {}
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors, None

    raise ValueError(f"Unsupported adapter file type: {input_path.name}")


def partition_lora_adapter_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Keep only keys ending with standard LoRA A/B weight suffixes."""
    kept: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in tensors.items():
        if key.endswith(LORA_ADAPTER_KEY_SUFFIXES):
            kept[key] = value
        else:
            dropped.append(key)
    return kept, dropped


def warn_ignored_non_lora_keys(
    dropped_keys: list[str],
    *,
    show_limit: int,
) -> None:
    if not dropped_keys:
        return
    n = len(dropped_keys)
    print(
        "\nWarning: Ignoring tensor(s) whose names do not end with "
        f"{LORA_ADAPTER_KEY_SUFFIXES[0]!r} or {LORA_ADAPTER_KEY_SUFFIXES[1]!r} "
        f"({n} total):"
    )
    limit = max(show_limit, 0)
    for key in dropped_keys[:limit]:
        print(f"  - {key}")
    if n > limit:
        print(f"  ... and {n - limit} more")


def save_tensors(
    output_path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> None:
    suffix = output_path.suffix
    if suffix == ".safetensors":
        safetensors_save_file(tensors, str(output_path), metadata=metadata)
        return
    if suffix in {".bin", ".pt"}:
        torch.save(tensors, str(output_path))
        return
    raise ValueError(f"Unsupported output file type: {output_path.name}")


def rename_keys(
    tensors: dict[str, torch.Tensor],
    old_prefix: str,
    new_prefix: str,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]]]:
    renamed: dict[str, torch.Tensor] = {}
    changed_pairs: list[tuple[str, str]] = []

    for key, value in tensors.items():
        new_key = key.replace(old_prefix, new_prefix, 1) if key.startswith(old_prefix) else key
        if new_key in renamed:
            raise ValueError(
                "Key collision after rename: "
                f"'{key}' -> '{new_key}', but '{new_key}' already exists."
            )
        renamed[new_key] = value
        if new_key != key:
            changed_pairs.append((key, new_key))

    return renamed, changed_pairs


def normalize_adapter_config_for_vllm(
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Clear ``modules_to_save`` when non-empty; vLLM only supports it empty."""
    modified = False
    if "modules_to_save" not in config:
        return config, modified
    modules_to_save = config["modules_to_save"]
    if modules_to_save in (None, [], ()):
        return config, modified
    config["modules_to_save"] = []
    modified = True
    return config, modified


def copy_adapter_config_for_output(
    lora_dir: Path,
    output_adapter_path: Path,
    *,
    dry_run: bool,
) -> None:
    """Copy ``adapter_config.json`` from ``lora_dir`` beside ``output_adapter_path``."""
    src = lora_dir / "adapter_config.json"
    if not src.is_file():
        print(
            f"Warning: adapter_config.json not found under {lora_dir}; "
            "skipping config copy."
        )
        return

    dst = output_adapter_path.parent / "adapter_config.json"
    with open(src, encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(
            "adapter_config.json must contain a JSON object, "
            f"got {type(loaded).__name__}"
        )

    config, cleared_modules = normalize_adapter_config_for_vllm(loaded)

    if dry_run:
        print(f"\nDry-run: would write adapter config to {dst} (from {src})")
        if cleared_modules:
            print(
                "  (would clear non-empty modules_to_save for vLLM compatibility)"
            )
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote adapter config: {dst}")
    if cleared_modules:
        print("  Cleared modules_to_save for vLLM compatibility.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename Qwen3 LoRA tensor prefixes for Omni Thinker language_model.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        required=True,
        help="LoRA adapter directory containing adapter_model.*",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit input file path. Defaults to auto-detect in --lora-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted weights "
            f"({DEFAULT_OUTPUT_ADAPTER_FILENAME}) and adapter_config.json. "
            "If omitted, write to '<input>.renamed<suffix>' next to the input file."
        ),
    )
    parser.add_argument(
        "--old-prefix",
        type=str,
        default="base_model.model.model.",
        help="Key prefix to replace.",
    )
    parser.add_argument(
        "--new-prefix",
        type=str,
        default="base_model.model.language_model.model.",
        help="Replacement key prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview key changes without writing files.",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=30,
        help="Max number of changed key pairs to print.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input if args.input is not None else detect_adapter_file(args.lora_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input adapter file not found: {input_path}")

    if args.output_dir is not None:
        return input_path, args.output_dir / DEFAULT_OUTPUT_ADAPTER_FILENAME

    return input_path, input_path.with_name(
        f"{input_path.stem}.renamed{input_path.suffix}"
    )


def main() -> None:
    args = parse_args()
    input_path, output_path = resolve_paths(args)

    tensors, metadata = load_tensors(input_path)
    total_keys = len(tensors)
    tensors, dropped_keys = partition_lora_adapter_tensors(tensors)

    if not tensors:
        if total_keys == 0:
            raise ValueError(f"No tensors found in adapter file: {input_path}")
        raise ValueError(
            "All tensors were ignored: none end with "
            f"{LORA_ADAPTER_KEY_SUFFIXES[0]!r} or {LORA_ADAPTER_KEY_SUFFIXES[1]!r}."
        )

    renamed, changed_pairs = rename_keys(tensors, args.old_prefix, args.new_prefix)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Tensor keys in checkpoint: {total_keys}")
    if dropped_keys:
        print(
            f"LoRA tensors after filter: {len(tensors)} "
            f"(ignored {len(dropped_keys)} non-LoRA key(s))"
        )
    else:
        print(f"LoRA tensors: {len(tensors)}")
    warn_ignored_non_lora_keys(dropped_keys, show_limit=args.show_limit)
    print(f"Renamed keys: {len(changed_pairs)}")

    if changed_pairs:
        print("\nSample key changes:")
        for old_key, new_key in changed_pairs[: max(args.show_limit, 0)]:
            print(f"- {old_key} -> {new_key}")
        if len(changed_pairs) > args.show_limit:
            hidden_count = len(changed_pairs) - args.show_limit
            print(f"... and {hidden_count} more")
    else:
        print("No keys matched the provided --old-prefix.")

    if args.dry_run:
        print("\nDry-run enabled; no file written.")
        copy_adapter_config_for_output(args.lora_dir, output_path, dry_run=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tensors(output_path, renamed, metadata)
    copy_adapter_config_for_output(args.lora_dir, output_path, dry_run=False)
    print("Done.")


if __name__ == "__main__":
    main()

