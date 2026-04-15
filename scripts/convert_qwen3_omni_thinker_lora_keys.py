#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batch convert LoRA tensor keys for Qwen3 Omni Thinker language_model loading.

Typical use case:
- LoRA was trained against Qwen3MoeForCausalLM and keys look like:
    base_model.model.model.layers.0....
- Qwen3OmniMoeThinkerForConditionalGeneration expects language model keys like:
    base_model.model.language_model.model.layers.0....

This script rewrites key prefixes in adapter weight files and writes the
standardised output (always ``adapter_model.safetensors``) together with a
normalised ``adapter_config.json`` into the **required** ``--output-dir``.
``--output-dir`` must differ from ``--lora-dir`` to prevent accidental
overwrites.  Non-empty ``modules_to_save`` in the config is cleared (vLLM
requires it empty).

It supports two old-prefix LoRA key layouts and auto-detects the input layout:
1) fused expert format:
   base_model.model.model.layers.0.mlp.up_proj.lora_A.weight
2) per expert format:
   base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_A.weight

You can explicitly choose the output LoRA layout with ``--target-lora-format``.
Default output is per expert format.

Tensor keys that do not end with ``.lora_A.weight`` or ``.lora_B.weight`` are
treated as invalid for this adapter and are omitted from the output after a
warning.

Supported input files:
- adapter_model.safetensors (preferred)
- adapter_model.bin
- adapter_model.pt

Output is always ``adapter_model.safetensors`` (safetensors format).

Examples:
    # Preview changes only
    python scripts/convert_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora --output-dir /path/to/output --dry-run

    # Convert with auto-detected prefix and default per_expert format
    python scripts/convert_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora --output-dir /path/to/output

    # Force output to fused expert format
    python scripts/convert_qwen3_omni_thinker_lora_keys.py \
      --lora-dir /path/to/lora --output-dir /path/to/output \
      --target-lora-format fused_expert
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

# Default output weights filename when using --output-dir (always safetensors).
DEFAULT_OUTPUT_ADAPTER_FILENAME = "adapter_model.safetensors"

# Standard PEFT LoRA weight suffixes; other tensors are warned and skipped.
LORA_ADAPTER_KEY_SUFFIXES = (".lora_A.weight", ".lora_B.weight")

LORA_FORMAT_FUSED_EXPERT = "fused_expert"
LORA_FORMAT_PER_EXPERT = "per_expert"
LORA_FORMAT_CHOICES = (LORA_FORMAT_FUSED_EXPERT, LORA_FORMAT_PER_EXPERT)
DEFAULT_TARGET_LORA_FORMAT = LORA_FORMAT_PER_EXPERT
DEFAULT_FUSED_TO_PER_EXPERT_INDEX = 0
EXPERT_PROJECTION_NAME_PATTERN = r"(?:up_proj|down_proj|gate_proj)"
SUPPORTED_LORA_PREFIXES = (
    # qwen3_moe peft layout
    "base_model.model.model.",
    # default vllm layout
    "base_model.model.language_model.model.",
    # ms-swift qwen3_omni peft layout
    "base_model.model.thinker.model.",
)
LORA_PREFIX_PATTERN = (
    r"base_model\.model\."
    r"(?:model|language_model\.model|thinker\.model)\."
)

FUSED_EXPERT_OLD_PREFIX_PATTERN = re.compile(
    r"^" + LORA_PREFIX_PATTERN + r"layers\.\d+\.mlp\."
    + EXPERT_PROJECTION_NAME_PATTERN
    + r"\.lora_[AB]\.weight$"
)
PER_EXPERT_OLD_PREFIX_PATTERN = re.compile(
    r"^" + LORA_PREFIX_PATTERN + r"layers\.\d+\.mlp\.experts\.\d+\."
    + EXPERT_PROJECTION_NAME_PATTERN
    + r"\.lora_[AB]\.weight$"
)

FUSED_TO_PER_REWRITE_PATTERN = re.compile(
    r"^(?P<head>.*\.layers\.\d+\.mlp\.)"
    r"(?P<tail>"
    + EXPERT_PROJECTION_NAME_PATTERN
    + r"\.lora_[AB]\.weight)$"
)
PER_TO_FUSED_REWRITE_PATTERN = re.compile(
    r"^(?P<head>.*\.layers\.\d+\.mlp\.)experts\.\d+\."
    r"(?P<tail>"
    + EXPERT_PROJECTION_NAME_PATTERN
    + r"\.lora_[AB]\.weight)$"
)


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
        tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] | None = None
        with safe_open(str(input_path), framework="pt") as reader:
            metadata = reader.metadata()
            tensor_keys = reader.keys()
            for key in tensor_keys:
                tensors[key] = reader.get_tensor(key)
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


def print_lora_tensor_prefixes(
    raw_keys: Iterable[str],
    *,
    layer_index: int = 0,
    show_limit: int,
) -> None:
    """Print unique raw tensor prefixes for a given layer (prefer LoRA keys)."""
    layer_tag = f".layers.{layer_index}."
    lora_prefixes: list[str] = []
    seen_prefixes: set[str] = set()
    non_lora_prefixes: list[str] = []
    seen_non_lora: set[str] = set()

    for key in raw_keys:
        if layer_tag not in key:
            continue
        if key.endswith(LORA_ADAPTER_KEY_SUFFIXES):
            prefix = re.sub(r"\.lora_[AB]\.weight$", "", key)
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                lora_prefixes.append(prefix)
            continue

        non_lora_prefix = key.rsplit(".", 1)[0] if "." in key else key
        if non_lora_prefix not in seen_non_lora:
            seen_non_lora.add(non_lora_prefix)
            non_lora_prefixes.append(non_lora_prefix)

    print(f"\nRaw LoRA tensor prefixes (layer {layer_index}):")
    limit = max(show_limit, 0)
    if lora_prefixes:
        for prefix in lora_prefixes[:limit]:
            print(f"  - {prefix}")
        if len(lora_prefixes) > limit:
            print(f"  ... and {len(lora_prefixes) - limit} more")
        return

    print(f"  (no layer {layer_index} LoRA tensor prefixes found)")
    if not non_lora_prefixes:
        print(
            f"  (no layer {layer_index} tensor prefixes "
            "found in raw checkpoint)"
        )
        return

    print(f"  Raw non-LoRA tensor prefixes (layer {layer_index}):")
    for prefix in non_lora_prefixes[:limit]:
        print(f"    - {prefix}")
    if len(non_lora_prefixes) > limit:
        print(f"    ... and {len(non_lora_prefixes) - limit} more")


def is_fused_expert_old_prefix_lora_key(key: str) -> bool:
    return bool(FUSED_EXPERT_OLD_PREFIX_PATTERN.match(key))


def is_per_expert_old_prefix_lora_key(key: str) -> bool:
    return bool(PER_EXPERT_OLD_PREFIX_PATTERN.match(key))


def detect_old_prefix_lora_format(keys: Iterable[str]) -> str:
    fused_match_count = 0
    per_expert_match_count = 0
    for key in keys:
        if is_fused_expert_old_prefix_lora_key(key):
            fused_match_count += 1
        if is_per_expert_old_prefix_lora_key(key):
            per_expert_match_count += 1

    if fused_match_count > 0 and per_expert_match_count == 0:
        return LORA_FORMAT_FUSED_EXPERT
    if per_expert_match_count > 0 and fused_match_count == 0:
        return LORA_FORMAT_PER_EXPERT
    if fused_match_count == 0 and per_expert_match_count == 0:
        supported_prefix_text = ", ".join(
            repr(prefix) for prefix in SUPPORTED_LORA_PREFIXES
        )
        raise ValueError(
            "Unable to detect LoRA format from adapter keys. "
            "No keys matched fused-expert or per-expert patterns "
            f"under supported prefixes {supported_prefix_text}."
        )

    raise ValueError(
        "Ambiguous old-prefix LoRA format: both fused-expert and per-expert "
        "patterns were detected in the same adapter."
    )


def infer_old_prefix(
    keys: Iterable[str],
    supported_prefixes: tuple[str, ...] = SUPPORTED_LORA_PREFIXES,
) -> str:
    """Auto-detect the common key prefix from LoRA adapter keys.

    Examines each key to determine which of the ``supported_prefixes`` it
    starts with, then returns the unique matched prefix.  Raises if no keys
    match or if multiple distinct prefixes are detected.
    """
    prefix_counts: dict[str, int] = {p: 0 for p in supported_prefixes}
    unmatched = 0

    for key in keys:
        for prefix in supported_prefixes:
            if key.startswith(prefix):
                prefix_counts[prefix] += 1
                break
        else:
            unmatched += 1

    found = {p: c for p, c in prefix_counts.items() if c > 0}

    if not found:
        supported_text = ", ".join(repr(p) for p in supported_prefixes)
        raise ValueError(
            "Unable to auto-detect --old-prefix from adapter keys. "
            f"No LoRA keys matched any supported prefix: {supported_text}."
        )

    if len(found) > 1:
        detail = ", ".join(f"{p!r}: {c} keys" for p, c in found.items())
        raise ValueError(
            f"Ambiguous key prefixes detected ({detail}). "
            "Please specify --old-prefix explicitly."
        )

    return next(iter(found))


def convert_single_lora_key_format(
    key: str,
    *,
    source_format: str,
    target_format: str,
    fused_to_per_expert_index: int,
) -> str:
    if source_format == target_format:
        return key

    if source_format == LORA_FORMAT_FUSED_EXPERT and target_format == LORA_FORMAT_PER_EXPERT:
        match = FUSED_TO_PER_REWRITE_PATTERN.match(key)
        if not match:
            return key
        return (
            f"{match.group('head')}experts.{fused_to_per_expert_index}."
            f"{match.group('tail')}"
        )

    if source_format == LORA_FORMAT_PER_EXPERT and target_format == LORA_FORMAT_FUSED_EXPERT:
        match = PER_TO_FUSED_REWRITE_PATTERN.match(key)
        if not match:
            return key
        return f"{match.group('head')}{match.group('tail')}"

    raise ValueError(
        f"Unsupported LoRA format conversion: {source_format} -> {target_format}"
    )


def convert_lora_key_format(
    tensors: dict[str, torch.Tensor],
    *,
    source_format: str,
    target_format: str,
    fused_to_per_expert_index: int,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]]]:
    converted: dict[str, torch.Tensor] = {}
    changed_pairs: list[tuple[str, str]] = []

    if source_format == LORA_FORMAT_FUSED_EXPERT and target_format == LORA_FORMAT_PER_EXPERT:
        detected_expert_count: int | None = None
        for key, value in tensors.items():
            match = FUSED_TO_PER_REWRITE_PATTERN.match(key)
            if not match:
                if key in converted:
                    raise ValueError(
                        "Key collision after LoRA format conversion: "
                        f"'{key}' already exists."
                    )
                converted[key] = value
                continue

            if value.ndim < 3:
                raise ValueError(
                    "Cannot split fused-expert tensor to per-expert layout: "
                    f"'{key}' has shape {tuple(value.shape)}. "
                    "Expected expert dimension at axis 0 (ndim >= 3)."
                )

            current_expert_count = int(value.shape[0])
            if current_expert_count <= 0:
                raise ValueError(
                    "Invalid fused-expert tensor with non-positive expert count: "
                    f"'{key}' has shape {tuple(value.shape)}."
                )
            if detected_expert_count is None:
                detected_expert_count = current_expert_count
            elif detected_expert_count != current_expert_count:
                raise ValueError(
                    "Inconsistent expert dimension across fused-expert tensors: "
                    f"expected {detected_expert_count}, got {current_expert_count} "
                    f"for key '{key}'."
                )

            for expert_index in range(current_expert_count):
                new_key = (
                    f"{match.group('head')}experts.{expert_index}.{match.group('tail')}"
                )
                if new_key in converted:
                    raise ValueError(
                        "Key collision after LoRA format conversion: "
                        f"'{key}' -> '{new_key}', but '{new_key}' already exists."
                    )
                converted[new_key] = value[expert_index].contiguous()
                changed_pairs.append((key, new_key))

        return converted, changed_pairs

    for key, value in tensors.items():
        new_key = convert_single_lora_key_format(
            key,
            source_format=source_format,
            target_format=target_format,
            fused_to_per_expert_index=fused_to_per_expert_index,
        )
        if new_key in converted:
            raise ValueError(
                "Key collision after LoRA format conversion: "
                f"'{key}' -> '{new_key}', but '{new_key}' already exists. "
                "This typically means multiple per-expert keys collapse into one "
                "fused key."
            )
        converted[new_key] = value
        if key != new_key:
            changed_pairs.append((key, new_key))

    return converted, changed_pairs


def save_tensors(
    output_path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> None:
    safetensors_save_file(tensors, str(output_path), metadata=metadata)


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
        description="Convert Qwen3 LoRA tensor keys for Omni Thinker language_model.",
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
        required=True,
        help=(
            "Directory for converted weights "
            f"({DEFAULT_OUTPUT_ADAPTER_FILENAME}) and adapter_config.json. "
            "Must differ from --lora-dir."
        ),
    )
    parser.add_argument(
        "--old-prefix",
        type=str,
        default=None,
        help=(
            "Key prefix to replace. If omitted, auto-detected from adapter "
            "keys. Supported prefixes: "
            + ", ".join(repr(p) for p in SUPPORTED_LORA_PREFIXES)
            + "."
        ),
    )
    parser.add_argument(
        "--new-prefix",
        type=str,
        default="base_model.model.language_model.model.",
        help="Replacement key prefix.",
    )
    parser.add_argument(
        "--target-lora-format",
        type=str,
        choices=LORA_FORMAT_CHOICES,
        default=DEFAULT_TARGET_LORA_FORMAT,
        help=(
            "Output LoRA key layout. Input layout is auto-detected from --lora-dir. "
            f"Default: {DEFAULT_TARGET_LORA_FORMAT}."
        ),
    )
    parser.add_argument(
        "--fused-to-per-expert-index",
        type=int,
        default=DEFAULT_FUSED_TO_PER_EXPERT_INDEX,
        help=(
            "Expert index inserted when converting fused_expert -> per_expert. "
            f"Default: {DEFAULT_FUSED_TO_PER_EXPERT_INDEX}."
        ),
    )
    parser.add_argument(
        "--inspect-layer",
        type=int,
        default=0,
        help=(
            "Layer index whose raw tensor prefixes are "
            "printed for inspection. Default: 0."
        ),
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

    if args.output_dir.resolve() == args.lora_dir.resolve():
        raise ValueError(
            "--output-dir must differ from --lora-dir to avoid "
            "overwriting the original adapter files."
        )

    return input_path, args.output_dir / DEFAULT_OUTPUT_ADAPTER_FILENAME


def main() -> None:
    args = parse_args()
    if args.fused_to_per_expert_index < 0:
        raise ValueError("--fused-to-per-expert-index must be >= 0")

    input_path, output_path = resolve_paths(args)

    raw_tensors, metadata = load_tensors(input_path)
    total_keys = len(raw_tensors)
    print_lora_tensor_prefixes(
        raw_tensors.keys(),
        layer_index=args.inspect_layer,
        show_limit=args.show_limit,
    )

    tensors, dropped_keys = partition_lora_adapter_tensors(raw_tensors)

    if not tensors:
        if total_keys == 0:
            raise ValueError(f"No tensors found in adapter file: {input_path}")
        raise ValueError(
            "All tensors were ignored: none end with "
            f"{LORA_ADAPTER_KEY_SUFFIXES[0]!r} or {LORA_ADAPTER_KEY_SUFFIXES[1]!r}."
        )

    if args.old_prefix is None:
        old_prefix = infer_old_prefix(tensors.keys())
        print(f"Auto-detected --old-prefix: {old_prefix!r}")
    else:
        old_prefix = args.old_prefix
        print(f"Using user-specified --old-prefix: {old_prefix!r}")

    if old_prefix == args.new_prefix:
        print(
            f"Note: --old-prefix {old_prefix!r} equals --new-prefix; "
            "prefix rename step will be a no-op."
        )

    detected_lora_format = detect_old_prefix_lora_format(tensors.keys())
    format_converted, format_changed_pairs = convert_lora_key_format(
        tensors,
        source_format=detected_lora_format,
        target_format=args.target_lora_format,
        fused_to_per_expert_index=args.fused_to_per_expert_index,
    )
    renamed, prefix_changed_pairs = rename_keys(
        format_converted, old_prefix, args.new_prefix
    )

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
    print(f"Detected old-prefix LoRA format: {detected_lora_format}")
    print(f"Target LoRA format: {args.target_lora_format}")
    print(f"Converted keys by format: {len(format_changed_pairs)}")
    print(f"Renamed keys by prefix: {len(prefix_changed_pairs)}")

    if format_changed_pairs:
        print("\nSample LoRA format key changes:")
        for old_key, new_key in format_changed_pairs[: max(args.show_limit, 0)]:
            print(f"- {old_key} -> {new_key}")
        if len(format_changed_pairs) > args.show_limit:
            hidden_count = len(format_changed_pairs) - args.show_limit
            print(f"... and {hidden_count} more")

    if prefix_changed_pairs:
        print("\nSample prefix key changes:")
        for old_key, new_key in prefix_changed_pairs[: max(args.show_limit, 0)]:
            print(f"- {old_key} -> {new_key}")
        if len(prefix_changed_pairs) > args.show_limit:
            hidden_count = len(prefix_changed_pairs) - args.show_limit
            print(f"... and {hidden_count} more")
    elif not format_changed_pairs:
        print("No keys matched requested format conversion or --old-prefix.")

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
