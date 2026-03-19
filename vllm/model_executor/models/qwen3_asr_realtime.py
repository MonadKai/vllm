# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-ASR realtime model."""

import asyncio
from collections.abc import AsyncGenerator, Iterable, Mapping

import numpy as np
import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.engine.protocol import StreamingInput
from vllm.envs import VLLM_ENGINE_ITERATION_TIMEOUT_S
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsRealtime,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
    _ASR_TEXT_TAG,
    _get_feat_extract_output_lengths,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import _I, BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalKwargsOptionalItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

logger = init_logger(__name__)

# Default language name in generation prompt so model outputs only transcription
# (no repeated "language Chinese<asr_text>" prefix per chunk).
DEFAULT_REALTIME_LANGUAGE_NAME = "Chinese"

# Use a small segment size for low-latency streaming, with overlap at
# boundaries to improve accuracy (like voxtral_realtime).
SEGMENT_DURATION_S = 5.0

# Overlap at segment boundaries (like voxtral_realtime); larger overlap
# reduces 漏字 and 边界重复 at the cost of a bit more latency.
DEFAULT_LOOK_BACK_S = 0.5
DEFAULT_LOOK_AHEAD_S = 0.5

# Max tokens of previous transcript as context; larger window helps avoid
# dropping words at boundaries (漏字) and reduces boundary repetition.
MAX_CONTEXT_TOKENS = 32


class Qwen3ASRRealtimeBuffer:
    """Audio buffer for Qwen3-ASR realtime streaming (Voxtral-style).

    Prompt per segment = prefix_before + last MAX_CONTEXT_TOKENS transcript
    + prefix_after. Context is placed before the current audio in the user
    message (prefix order) so the model sees "previous transcript → new audio".
    Fixed frame_size with look_back/look_ahead overlap, leftover for sliding.
    """

    def __init__(
        self,
        sampling_rate: int,
        segment_duration_s: float,
        look_back_s: float,
        look_ahead_s: float,
        prompt_token_ids_before_context: list[int],
        prompt_token_ids_after_context: list[int],
        *,
        token_output_queue: asyncio.Queue[list[int]] | None = None,
    ):
        self._sampling_rate = sampling_rate
        self._segment_size = int(segment_duration_s * sampling_rate)
        self._look_back_samples = int(look_back_s * sampling_rate)
        self._look_ahead_samples = int(look_ahead_s * sampling_rate)
        self._overlap_samples = (
            self._look_back_samples + self._look_ahead_samples
        )
        self._frame_size = self._segment_size + self._overlap_samples
        self._stride = self._segment_size

        self._prompt_token_ids_before_context = list(
            prompt_token_ids_before_context
        )
        self._prompt_token_ids_after_context = list(
            prompt_token_ids_after_context
        )
        self._accumulated_output_tokens: list[int] = []
        self._last_yield_accumulated_len = -1

        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._leftover: np.ndarray | None = None
        self._token_output_queue = token_output_queue

    async def _wait_previous_segment_output(self, timeout_s: float = 10.0) -> None:
        """Wait until feed_tokens has drained token_output_queue.

        Connection puts once per delta; wait until queue is empty so
        accumulated has the full previous output before building next prompt.
        """
        if self._last_yield_accumulated_len < 0:
            return
        if self._token_output_queue is None:
            for _ in range(30):
                await asyncio.sleep(0)
            return
        elapsed = 0.0
        poll_s = 0.01
        while not self._token_output_queue.empty() and elapsed < timeout_s:
            await asyncio.sleep(poll_s)
            elapsed += poll_s

    async def append_audio(self, audio_array: np.ndarray | None) -> None:
        await self._audio_queue.put(audio_array)

    async def append_tokens(self, tokens: Iterable[int]) -> None:
        self._accumulated_output_tokens.extend(tokens)

    async def get_input_stream(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            audio_arrays: list[np.ndarray] = (
                [self._leftover] if self._leftover is not None else []
            )
            total_samples = 0
            if self._leftover is not None:
                total_samples = len(self._leftover)
            need = self._frame_size - total_samples
            if self._leftover is None:
                need -= self._look_back_samples

            while need > 0:
                arr = await self._audio_queue.get()
                if arr is None:
                    await self._wait_previous_segment_output()
                    full = np.concatenate(audio_arrays) if audio_arrays else None
                    if self._leftover is None and full is not None:
                        full = np.concatenate([
                            np.zeros(
                                self._look_back_samples,
                                dtype=np.float32,
                            ),
                            full,
                        ])
                    if full is not None and len(full) > 0:
                        context = self._accumulated_output_tokens[
                            -MAX_CONTEXT_TOKENS:
                        ]
                        prompt_token_ids = list(
                            self._prompt_token_ids_before_context
                        )
                        prompt_token_ids.extend(context)
                        prompt_token_ids.extend(
                            self._prompt_token_ids_after_context
                        )
                        yield StreamingInput(
                            TokensPrompt(
                                prompt_token_ids=prompt_token_ids,
                                multi_modal_data={"audio": full},
                            )
                        )
                    return
                audio_arrays.append(arr)
                total_samples += len(arr)
                need = self._frame_size - total_samples
                if self._leftover is None:
                    need -= self._look_back_samples

            await self._wait_previous_segment_output()

            if self._leftover is not None:
                concatenated = np.concatenate(audio_arrays)
            else:
                pad = np.zeros(
                    self._look_back_samples,
                    dtype=np.float32,
                )
                concatenated = np.concatenate([pad] + audio_arrays)

            frame = concatenated[: self._frame_size].copy()
            self._leftover = concatenated[self._stride :].copy()

            self._last_yield_accumulated_len = len(self._accumulated_output_tokens)
            context = self._accumulated_output_tokens[-MAX_CONTEXT_TOKENS:]
            prompt_token_ids = list(self._prompt_token_ids_before_context)
            prompt_token_ids.extend(context)
            prompt_token_ids.extend(self._prompt_token_ids_after_context)
            yield StreamingInput(
                TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": frame},
                )
            )


class Qwen3ASRRealtimeMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=None)
        tokenizer = self.info.get_tokenizer()
        self._audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {len(audios)}"
        )

        audio_data = audios[0]
        audio_feature_lengths = audio_data.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            if isinstance(audio_feature_lengths.data, torch.Tensor):
                audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths.data
                ).item()
            else:
                audio_len = int(
                    _get_feat_extract_output_lengths(
                        torch.tensor(audio_feature_lengths.data)
                    ).item()
                )
        else:
            audio_len = 0

        # Expand placeholder to audio_len pads so MRoPE seq_len matches.
        audio_pad_id = self._audio_pad_id

        # Find the audio_pad token position and expand it to audio_len tokens
        expanded_ids: list[int] = []
        pad_start_idx = -1
        for i, tid in enumerate(prompt_ids):
            if tid == audio_pad_id and pad_start_idx == -1:
                pad_start_idx = i
                expanded_ids.extend([audio_pad_id] * audio_len)
            else:
                expanded_ids.append(tid)

        if pad_start_idx == -1:
            pad_start_idx = 0

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=pad_start_idx,
            tokens=audio_len * [audio_pad_id],
            is_embed=None,
        )
        return expanded_ids, {"audio": [features_info]}


# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# endpoint needs a different processor (Qwen3ASRRealtimeMultiModalProcessor)
# than the base transcription endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRRealtimeMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
@support_torch_compile
class Qwen3ASRRealtimeGeneration(Qwen3ASRForConditionalGeneration, SupportsRealtime):
    # Allow enough tokens per ~3s chunk so sentences are not cut off mid-way.
    realtime_max_tokens = 256

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        audio_placeholder = cls.get_placeholder_str("audio", 0)
        # Put [Previous transcript:] + context before current audio to use it
        # as prefix; order: user → previous transcript → current audio → assistant.
        prompt_before_context = (
            "<|im_start|>user\n[Previous transcript:] "
        )
        prompt_after_context = (
            f"\n{audio_placeholder}\n<|im_end|>\n<|im_start|>assistant\n"
            f"language {DEFAULT_REALTIME_LANGUAGE_NAME}{_ASR_TEXT_TAG}"
        )
        prompt_token_ids_before = tokenizer.encode(prompt_before_context)
        prompt_token_ids_after = tokenizer.encode(prompt_after_context)

        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=SEGMENT_DURATION_S,
            look_back_s=DEFAULT_LOOK_BACK_S,
            look_ahead_s=DEFAULT_LOOK_AHEAD_S,
            prompt_token_ids_before_context=prompt_token_ids_before,
            prompt_token_ids_after_context=prompt_token_ids_after,
            token_output_queue=input_stream,
        )

        async def feed_audio() -> None:
            async for audio_chunk in audio_stream:
                await buffer.append_audio(audio_chunk)
            await buffer.append_audio(None)

        async def feed_tokens() -> None:
            while True:
                # DELTA mode: each put is the new token(s) only, not cumulative.
                delta_tokens = await asyncio.wait_for(
                    input_stream.get(),
                    timeout=VLLM_ENGINE_ITERATION_TIMEOUT_S,
                )
                await buffer.append_tokens(delta_tokens)

        audio_task = asyncio.create_task(feed_audio())
        token_task = asyncio.create_task(feed_tokens())

        try:
            async for streaming_input in buffer.get_input_stream():
                yield streaming_input.prompt
        finally:
            audio_task.cancel()
            token_task.cancel()

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        return SpeechToTextConfig(
            max_audio_clip_s=None,
            sample_rate=feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )
