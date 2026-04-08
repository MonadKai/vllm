# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
"""Parse Qwen3-ASR model text output (language + transcription).

Aligned with ``qwen_asr.inference.utils.parse_asr_output`` in the official
Qwen3-ASR package (simplified: no repetition repair).
"""

from __future__ import annotations

import re
from typing import List, Tuple

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _normalize_language_name(language: str) -> str:
    s = str(language).strip()
    if not s:
        return ""
    return s[:1].upper() + s[1:].lower()


def parse_qwen3_asr_output(
    raw: str | None, user_language: str | None = None
) -> tuple[str, str]:
    """Parse one Qwen3-ASR completion into ``(language, text)``.

    If ``user_language`` is set, the raw string is treated as plain text and
    language is forced to ``user_language``.
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    if user_language:
        return user_language, s

    has_tag = _ASR_TEXT_TAG in s
    if not has_tag:
        return "", s.strip()

    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    meta_lower = meta_part.lower()

    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        return "", t

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX) :].strip()
            if val:
                lang = _normalize_language_name(val)
            break

    return lang, text_part.strip()


def split_qwen3_asr_concatenated(raw: str | None) -> List[Tuple[str, str]]:
    """Split concatenated per-segment ``language …<asr_text>…`` outputs.

    Falls back to :func:`parse_qwen3_asr_output` when there is no repeatable
    multi-segment pattern.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []

    if _ASR_TEXT_TAG not in s:
        lang, text = parse_qwen3_asr_output(s)
        return [(lang, text)]

    # Repeated segments: language X<asr_text>TEXT ... language Y<asr_text>...
    pattern = re.compile(
        r"language\s+(\S+?)\s*<asr_text>\s*(.*?)(?=language\s+\S+\s*<asr_text>|$)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(s))
    if matches:
        return [
            (_normalize_language_name(m.group(1)), m.group(2).strip()) for m in matches
        ]

    lang, text = parse_qwen3_asr_output(s)
    return [(lang, text)]
