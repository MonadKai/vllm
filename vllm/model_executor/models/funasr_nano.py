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
"""Inference-only Funasr-Nano model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.funasr_nano import FunasrNanoFeatureExtractor
from transformers.models.funasr_nano import FunasrNanoProcessor
from transformers.models.funasr_nano import FunasrNanoSenseVoiceEncoder
from transformers.models.funasr_nano import FunasrNanoMultiModalProjector
from transformers.models.funasr_nano import FunasrNanoSenseVoiceConfig, FunasrNanoConfig

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
    Computes the output length of the funasr nano sensevoice encoder
    """
    return input_lengths, input_lengths


# # === Audio Inputs === #
class FunasrNanoFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
    """
    type: Literal["audio_features"]
    input_features: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("na", "nmb", 560),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", "nmb"),
    ]


class FunasrNanoEmbeddingInputs(TensorSchema):
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


FunasrNanoInputs = Union[FunasrNanoFeatureInputs, FunasrNanoEmbeddingInputs]


# === Audio Encoder === #


class FunasrNanoProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(FunasrNanoConfig)

    def get_hf_processor(
        self,
        **kwargs: object
    ) -> FunasrNanoProcessor:
        return self.ctx.get_hf_processor(FunasrNanoProcessor, **kwargs)

    def get_feature_extractor(
        self,
        **kwargs: object
    ) -> FunasrNanoFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, FunasrNanoFeatureExtractor)
        return feature_extractor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int = None,
        mm_counts: Mapping[str, int] = None,
    ) -> Mapping[str, int]:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor
        return {"audio": feature_extractor.max_feature_length}

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class FunasrNanoDummyInputsBuilder(BaseDummyInputsBuilder[FunasrNanoProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token
        # assert audio_token == "<|vision_pad|>"
        audio_bos_token = hf_processor.audio_bos_token
        audio_eos_token = hf_processor.audio_eos_token
        return f"{audio_bos_token}{audio_token}{audio_eos_token}" * num_audios

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


def _funasr_nano_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


class FunasrNanoMultiModalDataParser(MultiModalDataParser):

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_funasr_nano_field_config,
            )

        return super()._parse_audio_data(data)


class FunasrNanoMultiModalProcessor(
    BaseMultiModalProcessor[FunasrNanoProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return FunasrNanoMultiModalDataParser(target_sr=feature_extractor.fs)

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
        return _funasr_nano_field_config(hf_inputs)

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
        audio_token = getattr(processor, "audio_token", "<|vision_pad|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|vision_start|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|vision_end|>")

        # audio_token_id = vocab[audio_token]
        audio_token_id = vocab["!"]
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

        def get_replacement_funasr_nano(item_idx: int):

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
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=f"{audio_bos_token}{audio_token}{audio_eos_token}",
                replacement=get_replacement_funasr_nano,
            )
        ]



@MULTIMODAL_REGISTRY.register_processor(
    FunasrNanoMultiModalProcessor,
    info=FunasrNanoProcessingInfo,
    dummy_inputs=FunasrNanoDummyInputsBuilder,
)
class FunasrNanoForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return  "<|vision_start|><|vision_pad|><|vision_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # HINT: funasr_nano is a mixed precision model
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

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

        with set_default_torch_dtype(self.audio_tower_dtype):
            self.audio_tower = FunasrNanoSenseVoiceEncoder(audio_config)
            if os.environ.get("VLLM_COMPILE_AUDIO_TOWER", "0") == "1":
                self.audio_tower.forward = torch.compile(self.audio_tower.forward)

        self.multi_modal_projector_dtype = self.audio_tower_dtype
        with set_default_torch_dtype(self.multi_modal_projector_dtype):
            self.multi_modal_projector = FunasrNanoMultiModalProjector(config)
            if os.environ.get("VLLM_COMPILE_MULTI_MODAL_PROJECTOR", "0") == "1":
                self.multi_modal_projector.forward = torch.compile(
                    self.multi_modal_projector.forward
                )

        self.quant_config = quant_config

        # HINT: language_model raw dtype is bfloat16
        text_config = config.text_config
        if hasattr(text_config, "torch_dtype"):
            self.language_model_dtype = text_config.torch_dtype
        else:
            self.language_model_dtype = text_config.dtype
        vllm_config.model_config.dtype = self.language_model_dtype
        with set_default_torch_dtype(self.language_model_dtype):
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
    ) -> Optional[FunasrNanoInputs]:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop('audio_embeds', None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")
            audio_embeds = self._validate_and_reshape_mm_tensor(
                audio_embeds, "audio_embeds")
            return FunasrNanoEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            input_features = self._validate_and_reshape_mm_tensor(
                input_features, "input_features"
            )
            feature_attention_mask = self._validate_and_reshape_mm_tensor(
                feature_attention_mask, "feature_attention_mask"
            )
            return FunasrNanoFeatureInputs(type="audio_features", input_features=input_features, feature_attention_mask=feature_attention_mask)
        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: FunasrNanoInputs
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

        selected_audio_feature, audio_output_lengths = self.audio_tower(
            input_features, audio_feature_lengths=audio_feat_lengths
        )

        selected_audio_feature = selected_audio_feature.to(self.multi_modal_projector_dtype)

        audio_features = self.multi_modal_projector(selected_audio_feature, audio_output_lengths)
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)