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
"""Inference-only Fun-Audio-Chat model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.models.funaudiochat import FunAudioChatProcessor
from transformers.models.funaudiochat import FunAudioChatConfig
from transformers.models.funaudiochat.modeling_funaudiochat import FunAudioChatAudioEncoder
from transformers.models.funaudiochat.modeling_funaudiochat import FunAudioChatDiscreteEncoder

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (AudioItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems)
from vllm.multimodal.parse import (AudioProcessorItems, DictEmbeddingItems,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)


def _get_feat_extract_output_lengths(
    input_lengths: torch.LongTensor
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Computes the output length of the funaudiochat audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


# # === Audio Inputs === #
class FunAudioChatFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - nst: Number of speech tokens
    """
    type: Literal["audio_features"]
    input_features: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("na", 128, "nmb"),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", "nmb"),
    ]

    speech_ids: Annotated[
        torch.Tensor,
        TensorShape("na", "nst"),
    ]

    speech_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", "nst"),
    ]

    feature_exist_mask: Annotated[
        torch.Tensor,
        TensorShape("na"),
    ]


class FunAudioChatEmbeddingInputs(TensorSchema):
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


FunAudioChatInputs = Union[FunAudioChatFeatureInputs, FunAudioChatEmbeddingInputs]


# === Audio Encoder === #


class FunAudioChatProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(FunAudioChatConfig)

    def get_hf_processor(
        self,
        **kwargs: object
    ) -> FunAudioChatProcessor:
        return self.ctx.get_hf_processor(FunAudioChatProcessor, **kwargs)

    def get_feature_extractor(
        self,
        **kwargs: object
    ) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int = None,
        mm_counts: Mapping[str, int] = None,
    ) -> Mapping[str, int]:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor
        return {"audio": 30000}

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class FunAudioChatDummyInputsBuilder(BaseDummyInputsBuilder[FunAudioChatProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token # "<|AUDIO|>"
        # assert audio_token == "<|AUDIO_PAD|>"
        audio_bos_token = hf_processor.audio_bos_token # "<|audio_bos|>"
        audio_eos_token = hf_processor.audio_eos_token # "<|audio_eos|>"
        return f"{audio_bos_token}{audio_token}{audio_eos_token}" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


def _funaudiochat_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        speech_ids=MultiModalFieldConfig.batched("audio"),
        speech_attention_mask=MultiModalFieldConfig.batched("audio"),
        feature_exist_mask=MultiModalFieldConfig.batched("audio"),
    )


class FunAudioChatMultiModalDataParser(MultiModalDataParser):

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_funaudiochat_field_config,
            )

        return super()._parse_audio_data(data)


class FunAudioChatMultiModalProcessor(
    BaseMultiModalProcessor[FunAudioChatProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return FunAudioChatMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

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
            sampling_rate=feature_extractor.sampling_rate,
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
        return _funaudiochat_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        hf_config = self.info.get_hf_config()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")

        audio_token_id = vocab[audio_token]
        audio_bos_id = vocab[audio_bos_token]
        audio_eos_id = vocab[audio_eos_token]

        # 获取 group_size 用于计算 DiscreteEncoder 的输出长度
        group_size = getattr(hf_config.audio_config, "group_size", 5)

        out_mm_data = out_mm_kwargs.get_data()
        # 使用 speech_attention_mask 计算输出长度（基于 DiscreteEncoder 的逻辑）
        speech_attention_mask = out_mm_data.get("speech_attention_mask")
        if speech_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(speech_attention_mask, torch.Tensor)
            speech_lengths = speech_attention_mask.sum(-1)
            # DiscreteEncoder 的输出长度计算: (input_lengths + group_size - 1) // group_size
            audio_output_lens = (speech_lengths + group_size - 1) // group_size
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_funaudiochat(item_idx: int):

            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape
                           ) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(f"The audio (len={audio_len}) is too short "
                                 "to be represented inside the model")

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                [audio_bos_id] + audio_tokens + [audio_eos_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=f"{audio_bos_token}{audio_token}{audio_eos_token}",
                replacement=get_replacement_funaudiochat,
            )
        ]



@MULTIMODAL_REGISTRY.register_processor(
    FunAudioChatMultiModalProcessor,
    info=FunAudioChatProcessingInfo,
    dummy_inputs=FunAudioChatDummyInputsBuilder,
)
class FunAudioChatForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return  "<|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        audio_config = config.audio_config
        if hasattr(audio_config, "torch_dtype"):
            self.audio_tower_dtype = audio_config.torch_dtype
        else:
            self.audio_tower_dtype = audio_config.dtype

        self.continuous_audio_tower = FunAudioChatAudioEncoder(audio_config)
        self.audio_tower = FunAudioChatDiscreteEncoder(audio_config)

        if os.environ.get("VLLM_COMPILE_AUDIO_TOWER", "0") == "1":
            self.continuous_audio_tower.forward = torch.compile(self.continuous_audio_tower.forward)
            self.audio_tower.forward = torch.compile(self.audio_tower.forward)

        self.quant_config = quant_config

        # HINT: language_model raw dtype is bfloat16
        text_config = config.text_config
        if hasattr(text_config, "torch_dtype"):
            self.language_model_dtype = text_config.torch_dtype
        else:
            self.language_model_dtype = text_config.dtype
        vllm_config.model_config.dtype = self.language_model_dtype
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return mm_input.reshape(-1, *mm_input.shape[2:])
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[FunAudioChatInputs]:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop('audio_embeds', None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        speech_ids = kwargs.pop("speech_ids", None)
        speech_attention_mask = kwargs.pop("speech_attention_mask", None)
        feature_exist_mask = kwargs.pop("feature_exist_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")
            audio_embeds = self._validate_and_reshape_mm_tensor(
                audio_embeds, "audio_embeds")
            return FunAudioChatEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            input_features = self._validate_and_reshape_mm_tensor(
                input_features, "input_features"
            )
            feature_attention_mask = self._validate_and_reshape_mm_tensor(
                feature_attention_mask, "feature_attention_mask"
            )
            speech_ids = self._validate_and_reshape_mm_tensor(
                speech_ids, "speech_ids"
            )
            speech_attention_mask = self._validate_and_reshape_mm_tensor(
                speech_attention_mask, "speech_attention_mask"
            )
            feature_exist_mask = self._validate_and_reshape_mm_tensor(
                feature_exist_mask, "feature_exist_mask"
            )
            return FunAudioChatFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                speech_ids=speech_ids,
                speech_attention_mask=speech_attention_mask,
                feature_exist_mask=feature_exist_mask,
            )
        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: FunAudioChatInputs
    ) -> tuple[torch.Tensor]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]
        speech_ids = audio_input["speech_ids"]
        speech_attention_mask = audio_input["speech_attention_mask"]
        feature_exist_mask = audio_input["feature_exist_mask"]

        # 计算音频长度
        audio_feature_lengths = feature_attention_mask.sum(-1)

        # 使用 continuous_audio_tower 的方法计算输出长度
        audio_feat_lengths, continuous_audio_output_lengths = (
            self.continuous_audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths
            )
        )

        # 使用 audio_tower（DiscreteEncoder）的方法计算输出长度
        speech_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                speech_attention_mask.sum(-1)
            )
        )

        # 将 speech_ids 长度扩展到 group_size 的倍数
        group_size = self.audio_tower.group_size
        speech_padding_target_length = (
            (speech_ids.shape[-1] + group_size - 1) // group_size
        ) * group_size
        if speech_padding_target_length > speech_ids.shape[-1]:
            padding_length = speech_padding_target_length - speech_ids.shape[-1]
            speech_ids = torch.nn.functional.pad(
                speech_ids, (0, padding_length), value=self.config.audio_config.pad_token_id
            )

        # Step 1: 使用 continuous_audio_tower 处理 mel 特征（如果有 continuous 特征）
        continuous_audio_features = None
        if feature_exist_mask.any():
            # 转换 input_features 格式：从 (batch, mel_bins, seq_len) 提取有效部分
            input_features = input_features.to(self.audio_tower_dtype)
            input_features_flat = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)

            # 需要计算 speech_maxlen
            speech_maxlen = speech_ids.shape[-1]

            # 使用 continuous_audio_tower 处理 mel 特征
            audio_outputs = self.continuous_audio_tower(
                input_features_flat,
                feature_lens=audio_feature_lengths,
                aftercnn_lens=audio_feat_lengths,
                speech_maxlen=speech_maxlen,
            )
            continuous_audio_features = audio_outputs.last_hidden_state

        # Step 2: 使用 audio_tower（DiscreteEncoder）处理 speech_ids，
        # 同时传入 continuous_audio_features 进行融合
        audio_features, *_ = self.audio_tower(
            speech_ids,
            continuous_audio_features=continuous_audio_features,
            continuous_audio_output_lengths=continuous_audio_output_lengths,
            feature_exist_mask=feature_exist_mask,
        )

        # 创建 mask 并提取有效特征
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = (
            torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
            < audio_output_lengths[:, None]
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
            t for t in masked_audio_features
        )
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
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
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_invert_tower."]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)