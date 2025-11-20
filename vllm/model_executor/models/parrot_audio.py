# SPDX-License-Identifier: Apache-2.0

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

import copy
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
from parrot_commons import _get_feat_extract_output_lengths
from parrot_commons.multi_modal_projector import LinearAdaptor
from parrot_commons.sense_voice_small import SenseVoiceEncoderSmall
from transformers import BatchFeature
from transformers.models.parrot_audio import ParrotAudioConfig
from transformers.models.parrot_audio import (
    ParrotAudioMultiModalProjector as TransformersParrotAudioMultiModalProjector,
)
from transformers.models.parrot_sensevoice import (
    ParrotSenseVoiceEncoder as TransformersParrotAudioEncoder,
)
from transformers.models.parrot_sensevoice import (
    ParrotSenseVoiceFeatureExtractor as ParrotAudioFeatureExtractor,
)
from transformers.models.parrot_sensevoice import (
    ParrotSenseVoiceProcessor as ParrotAudioProcessor,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
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
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)


def should_vllm_compile_mm_encoder(vllm_config: VllmConfig) -> bool:
    """Callable to be passed to `@support_torch_compile`'s `enable_if` argument."""
    return vllm_config.compilation_config.compile_mm_encoder


# # === Audio Inputs === #
class ParrotAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nmb", 560),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", 512),
    ]


class ParrotAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs"),
    ]


ParrotAudioInputs: TypeAlias = ParrotAudioFeatureInputs | ParrotAudioEmbeddingInputs


# === Audio Encoder === #


class ParrotAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(ParrotAudioConfig)

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> ParrotAudioProcessor:
        return self.ctx.get_hf_processor(ParrotAudioProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> ParrotAudioFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, ParrotAudioFeatureExtractor)
        return feature_extractor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor
        return {"audio": feature_extractor.max_feature_length}

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


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
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        # sampling_rate = feature_extractor.sampling_rate
        sampling_rate = feature_extractor.fs
        # audio_len = feature_extractor.chunk_length * sampling_rate
        audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


def _parrot_audio_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


class ParrotAudioMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_parrot_audio_field_config,
            )

        return super()._parse_audio_data(data)


class ParrotAudioMultiModalProcessor(
    BaseMultiModalProcessor[ParrotAudioProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return ParrotAudioMultiModalDataParser(target_sr=feature_extractor.fs)

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
        return _parrot_audio_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
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

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            mask_lengths = feature_attention_mask.sum(-1)
            _, audio_output_lens = _get_feat_extract_output_lengths(mask_lengths)

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_parrot_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

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
                replacement=get_replacement_parrot_audio,
            )
        ]


@support_torch_compile(
    dynamic_arg_dims={"input_features": 0, "audio_feature_lengths": 0},
    enable_if=should_vllm_compile_mm_encoder,
)
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
        attention_mask: torch.Tensor | None = None,  # None
        audio_feature_lengths: torch.Tensor | None = None,  # [16]
        head_mask: torch.Tensor | None = None,  # None
        output_attentions: bool | None = None,  # None
        output_hidden_states: bool | None = None,  # None
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs_pad, olens = self.sense_voice_small(
            input_features, ilens=audio_feature_lengths
        )  # xs_pad: [16, 500, 512], olens: [16]
        return xs_pad, olens


@support_torch_compile(
    dynamic_arg_dims={"audio_features": 0},
    enable_if=should_vllm_compile_mm_encoder,
)
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
    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|vision_start|>[FAKE_AUDIO]<|vision_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # HINT: parrot_audio is a mixed precision model
        from vllm.utils.torch_utils import set_default_torch_dtype

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.vllm_config = vllm_config
        mm_encoder_vllm_config = copy.deepcopy(vllm_config)
        if (
            mm_encoder_vllm_config.compilation_config.mode
            == CompilationMode.VLLM_COMPILE
        ):
            raise ValueError(
                "VLLM_COMPILE is not supported for parrot_audio's mm_encoder, please use DYNAMO_TRACE_ONCE or STOCK_TORCH_COMPILE instead"
            )
        self.mm_encoder_vllm_config = mm_encoder_vllm_config

        self.multimodal_config = multimodal_config

        self.audio_tower_dtype = config.audio_config.torch_dtype
        with set_default_torch_dtype(self.audio_tower_dtype):
            if os.environ.get("VLLM_USE_TRANSFORMERS_AUDIO_ENCODER", "0") == "1":
                self.audio_tower = TransformersParrotAudioEncoder(config.audio_config)
            else:
                self.audio_tower = ParrotAudioEncoder(vllm_config=vllm_config)

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

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> ParrotAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return ParrotAudioEmbeddingInputs(
                type="audio_embeds", audio_embeds=audio_embeds
            )

        if input_features is not None:
            return ParrotAudioFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: ParrotAudioInputs
    ) -> tuple[torch.Tensor]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )

        input_features = input_features.to(self.audio_tower_dtype)

        with set_forward_context(None, self.mm_encoder_vllm_config):
            audio_outputs = self.audio_tower(
                input_features, audio_feature_lengths=audio_feat_lengths
            )
        selected_audio_feature = audio_outputs[0]

        selected_audio_feature = selected_audio_feature.to(
            self.multi_modal_projector_dtype
        )

        with set_forward_context(None, self.mm_encoder_vllm_config):
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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        # HINT: convert to input_embeds dtype
        masked_audio_features = tuple(
            t.to(self.language_model_dtype) for t in masked_audio_features
        )
        return masked_audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
