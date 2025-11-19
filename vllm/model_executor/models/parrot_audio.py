# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Bairong Inc.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Parrot-Audio model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature
from transformers.models.parrot_audio import ParrotAudioConfig
from transformers.models.parrot_sensevoice import ParrotSenseVoiceProcessor as ParrotAudioProcessor
from transformers.models.parrot_sensevoice import ParrotSenseVoiceFeatureExtractor as ParrotAudioFeatureExtractor
from transformers.models.parrot_sensevoice import ParrotSenseVoiceEncoder as TransformersParrotAudioEncoder
from transformers.models.parrot_audio import (
    ParrotAudioMultiModalProjector as TransformersParrotAudioMultiModalProjector,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from parrot_commons import _get_feat_extract_output_lengths
from parrot_commons.multi_modal_projector import LinearAdaptor
from parrot_commons.sense_voice_small import SenseVoiceEncoderSmall


class ParrotAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(num_audios, num_mel_bins, 3000)`"""

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`"""


class ParrotAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(ParrotAudioConfig)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> ParrotAudioProcessor:
        return self.ctx.get_hf_processor(ParrotAudioProcessor, **kwargs)

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> ParrotAudioFeatureExtractor:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, ParrotAudioFeatureExtractor)
        return feature_extractor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int = None,
        mm_counts: Mapping[str, int] = None,
    ) -> Mapping[str, int]:
        max_output_lengths = 500
        return {"audio": max_output_lengths}

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        max_output_lengths = 500
        return {"audio": max_output_lengths}


class ParrotAudioDummyInputsBuilder(BaseDummyInputsBuilder[ParrotAudioProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token
        # assert audio_token == "[FAKE_AUDIO]"
        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        # sampling_rate = feature_extractor.sampling_rate
        sampling_rate = feature_extractor.fs
        # audio_len = feature_extractor.chunk_length * sampling_rate
        audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class ParrotAudioMultiModalProcessor(
    BaseMultiModalProcessor[ParrotAudioProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        # return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)
        return MultiModalDataParser(target_sr=feature_extractor.fs)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # NOTE - we rename audios -> audio in mm data because transformers has
        # deprecated audios for the qwen2audio processor and will remove
        # support for it in transformers 4.54.
        audios = mm_data.get("audios", [])
        if audios:
            mm_data["audio"] = audios

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            # sampling_rate=feature_extractor.sampling_rate,
            sampling_rate=feature_extractor.fs,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "[FAKE_AUDIO]")
        # HINT: audio model use vision token ???
        audio_bos_token = getattr(processor, "audio_bos_token", "<|vision_start|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|vision_end|>")

        audio_token_id = vocab[audio_token]
        audio_bos_id = vocab[audio_bos_token]
        audio_eos_id = vocab[audio_eos_token]

        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            if isinstance(feature_attention_mask, torch.Tensor):
                mask_lengths = feature_attention_mask.sum(-1)
            elif isinstance(feature_attention_mask, list):
                mask_lengths = torch.tensor(
                    [i.sum(-1).item() for i in feature_attention_mask]
                )
            _, audio_output_lens = _get_feat_extract_output_lengths(mask_lengths)

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_qwen2_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(
                    f"The audio (len={audio_len}) is too short "
                    "to be represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                [audio_bos_id] + audio_tokens + [audio_eos_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            )
        ]


# @support_torch_compile(dynamic_arg_dims={"input_features": 0, "audio_feature_lengths": 0})
class ParrotAudioEncoder(nn.Module):
    """
    ParrotSenseVoiceEncoder

    Args:
        config: ParrotSenseVoiceConfig
    """

    # Ignore copy
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        self.sense_voice_small = SenseVoiceEncoderSmall(
            input_size=audio_config.input_size,
            output_size=audio_config.output_size,
            attention_heads=audio_config.attention_heads,
            linear_units=audio_config.linear_units,
            num_blocks=audio_config.num_blocks,
            tp_blocks=audio_config.tp_blocks,
            dropout_rate=audio_config.dropout_rate,
            attention_dropout_rate=audio_config.attention_dropout_rate,
            normalize_before=audio_config.normalize_before,
            kernel_size=audio_config.kernel_size,
            sanm_shfit=audio_config.sanm_shfit,
        )

    def forward(
        self,
        input_features: torch.Tensor,  # [16, 500, 560]
        attention_mask: Optional[torch.Tensor] = None,  # None
        audio_feature_lengths: Optional[torch.Tensor] = None,  # [16]
        head_mask: Optional[torch.Tensor] = None,  # None
        output_attentions: Optional[bool] = None,  # None
        output_hidden_states: Optional[bool] = None,  # None
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs_pad, olens = self.sense_voice_small(
            input_features, ilens=audio_feature_lengths
        )  # xs_pad: [16, 500, 512], olens: [16]
        return xs_pad, olens


# @support_torch_compile(dynamic_arg_dims={"audio_features": 0})
class ParrotAudioMultiModalProjector(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        text_config = config.text_config

        self.adaptor = LinearAdaptor(
            encoder_dim=audio_config.output_size,
            ffn_dim=config.adaptor_ffn_dim,
            llm_dim=text_config.hidden_size,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.adaptor(audio_features)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    ParrotAudioMultiModalProcessor,
    info=ParrotAudioProcessingInfo,
    dummy_inputs=ParrotAudioDummyInputsBuilder,
)
class ParrotAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return  "<|vision_start|>[FAKE_AUDIO]<|vision_end|>"

        raise ValueError("Only audio modality is supported")


    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # HINT: parrot_audio is a mixed precision model
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower_dtype = config.audio_config.torch_dtype
        with set_default_torch_dtype(self.audio_tower_dtype):
            if os.environ.get("VLLM_USE_TRANSFORMERS_AUDIO_ENCODER", "0") == "1":
                self.audio_tower = TransformersParrotAudioEncoder(config.audio_config)
            else:
                self.audio_tower = ParrotAudioEncoder(vllm_config=vllm_config)
            if os.environ.get("VLLM_COMPILE_AUDIO_TOWER", "0") == "1":
                self.audio_tower.forward = torch.compile(self.audio_tower.forward)

        self.multi_modal_projector_dtype = self.audio_tower_dtype
        with set_default_torch_dtype(self.audio_tower_dtype):
            if (
                os.environ.get("VLLM_USE_TRANSFORMERS_MULTI_MODAL_PROJECTOR", "0")
                == "1"
            ):
                self.multi_modal_projector = TransformersParrotAudioMultiModalProjector(
                    config
                )
            else:
                self.multi_modal_projector = ParrotAudioMultiModalProjector(
                    vllm_config=vllm_config
                )
            if os.environ.get("VLLM_COMPILE_MULTI_MODAL_PROJECTOR", "0") == "1":
                self.multi_modal_projector.forward = torch.compile(
                    self.multi_modal_projector.forward
                )

        self.quant_config = quant_config

        # HINT: language_model raw dtype is bfloat16
        vllm_config.model_config.dtype = config.text_config.torch_dtype
        self.language_model_dtype = config.text_config.torch_dtype
        with set_default_torch_dtype(self.language_model_dtype):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        # HINT: branch that never hit
        elif isinstance(mm_input[0], list):
            if mm_input[0][0].dtype == torch.float:
                pad_value = 0.0
            elif mm_input[0][0].dtype == torch.bool:
                pad_value = False
            elif mm_input[0][0].dtype == torch.int:
                pad_value = 0
            return pad_sequence(
                mm_input[0],
                batch_first=True,
                padding_side="right",
                padding_value=pad_value,
            )
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[ParrotAudioInputs]:
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, "input_features"
        )
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, "feature_attention_mask"
        )
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of audio input features. "
                f"Got type: {type(input_features)}"
            )
        return ParrotAudioInputs(
            input_features=input_features, feature_attention_mask=feature_attention_mask
        )

    def _process_audio_input(
        self, audio_input: ParrotAudioInputs
    ) -> tuple[torch.Tensor]:
        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )

        input_features = input_features.to(self.audio_tower_dtype)

        audio_outputs = self.audio_tower(
            input_features, audio_feature_lengths=audio_feat_lengths
        )
        selected_audio_feature = audio_outputs[0]

        selected_audio_feature = selected_audio_feature.to(self.multi_modal_projector_dtype)

        audio_features = self.multi_modal_projector(selected_audio_feature)
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_audios, max_audio_tokens)
            .to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        # Split to tuple of embeddings for individual audio input.
        return torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        # HINT: convert to input_embeds dtype
        masked_audio_features = tuple(
            t.to(self.language_model_dtype) for t in masked_audio_features
        )
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            multimodal_embeddings = [
                i.to(inputs_embeds.dtype) for i in multimodal_embeddings
            ]
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.audio_token_index,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
