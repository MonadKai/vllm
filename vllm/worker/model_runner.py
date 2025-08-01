# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import gc
import inspect
import itertools
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm.auto import tqdm

import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionState
from vllm.attention.backends.utils import CommonAttentionState
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             graph_capture)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.sampler import (Sampler, SamplerOutput,
                                                get_sampler)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import (DeviceMemoryProfiler, GiB_bytes, PyObjectCache,
                        async_tensor_h2d, flatten_2d_lists,
                        is_pin_memory_available, supports_dynamo,
                        weak_ref_tensor)
from vllm.worker.model_runner_base import (
    InputProcessingError, ModelRunnerBase, ModelRunnerInputBase,
    ModelRunnerInputBuilderBase, _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8

_NUM_WARMUP_ITERS = 2

TModelInputForGPU = TypeVar('TModelInputForGPU', bound="ModelInputForGPU")

# For now, bump up cache limits for recompilations during CUDA graph warmups.
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 128


@dataclass(frozen=True)
class ModelInputForGPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    token_types: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None
    virtual_engine: int = 0
    async_callback: Optional[Callable] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    previous_hidden_states: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "inputs_embeds": self.inputs_embeds,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForGPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForGPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)

    # Exclude `async_callback` to be able to pickle this object
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["async_callback"]
        return state

    # TODO: What happens when we depickle this object?
    # How can we update this callback to properly pass it to the engine?
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.update({'async_callback': None})


@dataclass(frozen=True)
class ModelInputForGPUWithSamplingMetadata(ModelInputForGPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "inputs_embeds": self.inputs_embeds,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForGPUBuilder(ModelRunnerInputBuilderBase[ModelInputForGPU]):
    """Build ModelInputForGPU from SequenceGroupMetadata."""

    # Note: ideally we would be using a dataclass(kw_only=True)
    # here, so that this can be subclassed easily,
    # but kw_only is not supported in python<3.10.
    class InterDataForSeqGroup:
        """Intermediate data for the current sequence group."""

        def simple_reinit(self):
            self.input_tokens[0].clear()  # type: ignore
            self.inputs_embeds = None  # type: ignore
            self.input_positions[0].clear()  # type: ignore
            self.token_types[0].clear()  # type: ignore
            self.mrope_input_positions = None  # type: ignore
            self.seq_lens[0] = 0  # type: ignore
            self.orig_seq_lens[0] = 0  # type: ignore
            self.prompt_lens[0] = 0  # type: ignore
            self.query_lens[0] = 0  # type: ignore
            self.context_lens[0] = 0  # type: ignore
            self.curr_sliding_window_blocks[0] = 0  # type: ignore
            self.lora_index_mapping.clear()  # type: ignore
            self.lora_prompt_mapping.clear()  # type: ignore
            self.lora_requests.clear()  # type: ignore

        def __init__(
            self,
            *,
            # From sequence group metadata.
            request_id: str,
            seq_ids: List[int],
            is_prompt: bool,
            block_tables: Optional[Dict[int, List[int]]],
            computed_block_nums: List[int],
            n_seqs: int = 0,

            # Input tokens and positions.
            input_tokens: Optional[List[List[int]]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            input_positions: Optional[List[List[int]]] = None,
            token_types: Optional[List[List[int]]] = None,
            mrope_input_positions: Optional[List[List[List[int]]]] = None,

            # The sequence length (may be capped to the sliding window).
            seq_lens: Optional[List[int]] = None,
            # The original sequence length (before applying sliding window).
            # This is used to compute slot mapping.
            orig_seq_lens: Optional[List[int]] = None,
            # This is used in the dual-chunk flash attention backend.
            prompt_lens: Optional[List[int]] = None,
            # The query length.
            query_lens: Optional[List[int]] = None,
            # The number of tokens that are already computed.
            context_lens: Optional[List[int]] = None,
            # The current sliding window block.
            curr_sliding_window_blocks: Optional[List[int]] = None,

            # LoRA inputs.
            lora_index_mapping: Optional[List[List[int]]] = None,
            lora_prompt_mapping: Optional[List[List[int]]] = None,
            lora_requests: Optional[Set[LoRARequest]] = None,

            # Multi-modal inputs.
            multi_modal_kwargs: Optional[MultiModalKwargs] = None,
            multi_modal_placeholder_maps: Optional[Dict[
                str, MultiModalPlaceholderMap]] = None,

            # Whether the prefix cache is hit (prefill only).
            prefix_cache_hit: bool = False,
            reinit: bool = False,
            reinit_use_defaults: bool = False,
            encoder_seq_len: int = 0,
        ):
            if reinit:
                assert len(self.seq_ids) == len(seq_ids)  # type: ignore
                for i, seq_id in enumerate(seq_ids):
                    self.seq_ids[i] = seq_id  # type: ignore
            else:
                self.seq_ids = seq_ids

            self.request_id = request_id
            self.is_prompt = is_prompt
            self.block_tables = block_tables
            self.computed_block_nums = computed_block_nums
            self.n_seqs = n_seqs
            self.encoder_seq_len = encoder_seq_len

            if reinit:
                if len(self.seq_ids) == 1 and reinit_use_defaults:
                    self.simple_reinit()
                else:
                    if input_tokens:
                        self.input_tokens = input_tokens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_tokens[seq_id].clear()

                    self.inputs_embeds = inputs_embeds

                    if input_positions:
                        self.input_positions = input_positions
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_positions[seq_id].clear()

                    if token_types:
                        self.token_types = token_types
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.token_types[seq_id].clear()

                    self.mrope_input_positions = None

                    if seq_lens:
                        self.seq_lens = seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.seq_lens[seq_id] = 0

                    if orig_seq_lens:
                        self.orig_seq_lens = orig_seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.orig_seq_lens[seq_id] = 0

                    if prompt_lens:
                        self.prompt_lens = prompt_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.prompt_lens[seq_id] = 0

                    if query_lens:
                        self.query_lens = query_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.query_lens[seq_id] = 0

                    if context_lens:
                        self.context_lens = context_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.context_lens[seq_id] = 0

                    if curr_sliding_window_blocks:
                        self.curr_sliding_window_blocks = \
                            curr_sliding_window_blocks
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.curr_sliding_window_blocks[seq_id] = 0

                    if lora_index_mapping:
                        self.lora_index_mapping = lora_index_mapping
                    else:
                        self.lora_index_mapping.clear()

                    if lora_prompt_mapping:
                        self.lora_prompt_mapping = lora_prompt_mapping
                    else:
                        self.lora_prompt_mapping.clear()

                    if lora_requests:
                        self.lora_requests = lora_requests
                    else:
                        self.lora_requests.clear()

            else:
                self.input_tokens = input_tokens or []
                self.inputs_embeds = inputs_embeds
                self.input_positions = input_positions or []
                self.token_types = token_types or []
                self.mrope_input_positions = mrope_input_positions or None
                self.seq_lens = seq_lens or []
                self.orig_seq_lens = orig_seq_lens or []
                self.prompt_lens = prompt_lens or []
                self.query_lens = query_lens or []
                self.context_lens = context_lens or []
                self.curr_sliding_window_blocks = \
                    curr_sliding_window_blocks or []

                self.lora_index_mapping = lora_index_mapping or []
                self.lora_prompt_mapping = lora_prompt_mapping or []
                self.lora_requests = lora_requests or set()

            self.multi_modal_kwargs = multi_modal_kwargs
            self.multi_modal_placeholder_maps = multi_modal_placeholder_maps
            self.prefix_cache_hit = prefix_cache_hit

            self.n_seqs = len(self.seq_ids)

            if not reinit:
                self.__post_init__()

        def __post_init__(self):
            self.n_seqs = len(self.seq_ids)

            self.input_tokens = [[] for _ in range(self.n_seqs)]
            self.input_positions = [[] for _ in range(self.n_seqs)]
            self.token_types = [[] for _ in range(self.n_seqs)]
            self.mrope_input_positions = None
            self.seq_lens = [0] * self.n_seqs
            self.orig_seq_lens = [0] * self.n_seqs
            self.prompt_lens = [0] * self.n_seqs
            self.query_lens = [0] * self.n_seqs
            self.context_lens = [0] * self.n_seqs
            self.curr_sliding_window_blocks = [0] * self.n_seqs

            self.lora_index_mapping = []
            self.lora_prompt_mapping = []

        def __repr__(self) -> str:
            return (f"InterDataForSeqGroup("
                    f"request_id={self.request_id}, "
                    f"seq_ids={self.seq_ids}, "
                    f"is_prompt={self.is_prompt}, "
                    f"block_tables={self.block_tables}, "
                    f"computed_block_nums={self.computed_block_nums}, "
                    f"n_seqs={self.n_seqs}, "
                    f"input_tokens={self.input_tokens}, "
                    f"inputs_embeds.shape="
                    f"{getattr(self.inputs_embeds, 'shape', None)}, "
                    f"input_positions={self.input_positions}, "
                    f"token_types={self.token_types}, "
                    f"mrope_input_positions={self.mrope_input_positions}, "
                    f"seq_lens={self.seq_lens}, "
                    f"orig_seq_lens={self.orig_seq_lens}, "
                    f"query_lens={self.query_lens}, "
                    f"context_lens={self.context_lens}, "
                    f"multi_modal_kwargs={self.multi_modal_kwargs}")

    def gen_inter_data_builder(self, num_seqs: int):
        return lambda: ModelInputForGPUBuilder.InterDataForSeqGroup(
            request_id="",
            seq_ids=[0] * num_seqs,
            is_prompt=True,
            block_tables=None,
            computed_block_nums=[])

    def init_cached_inter_data(self, *args, **kwargs):
        assert len(args) == 0
        assert "seq_ids" in kwargs
        seq_ids = kwargs["seq_ids"]
        num_seqs = len(seq_ids)

        # The inter-data cache is per model_runner
        inter_data_cache = self.runner.inter_data_cache
        if num_seqs not in inter_data_cache:
            inter_data_cache[num_seqs] = PyObjectCache(
                self.gen_inter_data_builder(num_seqs))

        obj = inter_data_cache[num_seqs].get_object()
        obj.__init__(*args, **kwargs)
        return obj

    def reset_cached_inter_data(self):
        for cache in self.runner.inter_data_cache.values():
            cache.reset()

    def __init__(self,
                 runner: "GPUModelRunnerBase",
                 finished_requests_ids: Optional[List[str]] = None):
        super().__init__()
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
            self._compute_for_sliding_window,
            self._compute_lora_input,
        ]
        # Compute functions for each sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_group_compute_fns = [
            self._compute_multi_modal_input,
        ]

        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.enable_lora = self.runner.lora_config is not None

        # Attention metadata inputs.
        if self.attn_backend is not None:
            # spec decode (e.g. Medusa) does not have atten backend
            self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
                weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)
        if self.sliding_window is not None:
            self.sliding_window_blocks = (
                self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = \
                self.sliding_window_blocks * self.block_size

    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        self.finished_requests_ids = finished_requests_ids

        # if the current batch is decode-only.
        # will be set to False if there is any non-decode request.
        self.decode_only = True

        # Intermediate data (data in CPU before going to GPU) for
        # the current sequence group.
        self.inter_data_list: List[
            ModelInputForGPUBuilder.InterDataForSeqGroup] = []

        self.attn_metadata_builder.prepare()

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int,
                      seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).

        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_len, context_len + token_chunk_size)
        elif self.runner.scheduler_config.is_multi_step or \
            self.runner.model_config.is_encoder_decoder:
            context_len = seq_len - 1
        else:
            context_len = seq_data.get_num_computed_tokens()

        # Compute tokens.
        if seq_data.prompt_embeds is None:
            tokens = seq_data.get_token_ids()[context_len:seq_len]
            prompt_embeds = None
        else:
            tokens = [0] * (seq_len - context_len)
            prompt_embeds = seq_data.get_token_embeddings(
            )[context_len:seq_len]

        token_types = seq_group_metadata.token_type_ids

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.prompt_lens[seq_idx] = seq_data.get_prompt_len()
        inter_data.context_lens[seq_idx] = context_len
        inter_data.input_tokens[seq_idx].extend(tokens)
        inter_data.inputs_embeds = prompt_embeds
        inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))
        inter_data.token_types[seq_idx].extend(
            token_types if token_types else [])
        inter_data.query_lens[seq_idx] = seq_len - context_len

        if seq_data.mrope_position_delta is not None:
            if inter_data.mrope_input_positions is None:
                inter_data.mrope_input_positions = [None] * inter_data.n_seqs

            inter_data.mrope_input_positions[
                seq_idx] = MRotaryEmbedding.get_next_input_positions(
                    seq_data.mrope_position_delta,
                    context_len,
                    seq_len,
                )

    def _compute_for_prefix_cache_hit(
            self, inter_data: InterDataForSeqGroup, seq_idx: int,
            seq_group_metadata: SequenceGroupMetadata):
        """Check if hit prefix cache (i.e., some blocks are already computed).
        If hit, update input tokens and positions to only compute the
        remaining blocks.
        """
        computed_block_nums = inter_data.computed_block_nums

        # Note that prefix caching does not support sliding window.
        prefix_cache_hit = (computed_block_nums is not None
                            and len(computed_block_nums) > 0
                            and self.sliding_window is None
                            and inter_data.is_prompt)
        inter_data.prefix_cache_hit = prefix_cache_hit

        if not prefix_cache_hit:
            return

        assert computed_block_nums is not None
        # The cache hit prompt tokens in this sequence. Note that
        # this may be larger than the sequence length if chunked
        # prefill is enabled.
        prefix_cache_len = len(computed_block_nums) * self.block_size
        seq_group_metadata.seq_data[inter_data.seq_ids[
            seq_idx]].update_num_cached_tokens(prefix_cache_len)

        # The number of so far computed prompt tokens in this sequence.
        context_len = inter_data.context_lens[seq_idx]
        # The total number of prompt tokens in this sequence.
        # When chunked prefill is enabled, this is the token number of
        # computed chunks + current chunk.
        seq_len = inter_data.seq_lens[seq_idx]
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass
        elif context_len < prefix_cache_len < seq_len:
            # Partial hit. Compute the missing part.
            uncomputed_start = prefix_cache_len - context_len
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][uncomputed_start:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            inter_data.token_types[seq_idx] = inter_data.token_types[seq_idx][
                uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len
        elif seq_len <= prefix_cache_len:
            # Full hit. Only compute the last token to avoid
            # erroneous behavior. FIXME: Ideally we should directly
            # mark all tokens as computed in the scheduler and do not
            # schedule this sequence, so this case should not happen.
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][-1:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.token_types[seq_idx] = inter_data.token_types[seq_idx][
                -1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1

    def _compute_for_sliding_window(self, inter_data: InterDataForSeqGroup,
                                    seq_idx: int,
                                    seq_group_metadata: SequenceGroupMetadata):
        """Update seq_len and curr_sliding_window_block for the given
        sequence data (only required by decoding) if sliding window is enabled.
        """
        curr_sliding_window_block = 0
        sliding_seq_len = inter_data.seq_lens[seq_idx]
        if not inter_data.is_prompt and self.sliding_window is not None:
            # TODO(sang): This is a hack to make sliding window work with
            # paged attn. We can remove it if we make paged attn kernel
            # to properly handle slinding window attn.
            curr_sliding_window_block = self.sliding_window_blocks
            # number of elements in last block
            suff_len = inter_data.seq_lens[seq_idx] % self.block_size
            sliding_seq_len = min(inter_data.seq_lens[seq_idx],
                                  self.block_aligned_sliding_window + suff_len)
            if suff_len > 0:
                curr_sliding_window_block += 1

        inter_data.curr_sliding_window_blocks[
            seq_idx] = curr_sliding_window_block
        inter_data.seq_lens[seq_idx] = sliding_seq_len

    def _compute_lora_input(self, inter_data: InterDataForSeqGroup,
                            seq_idx: int,
                            seq_group_metadata: SequenceGroupMetadata):
        """If LoRA is enabled, compute LoRA index and prompt mapping."""
        if not self.enable_lora:
            return

        lora_id = seq_group_metadata.lora_int_id
        if lora_id > 0:
            inter_data.lora_requests.add(seq_group_metadata.lora_request)
        query_len = inter_data.query_lens[seq_idx]
        inter_data.lora_index_mapping.append([lora_id] * query_len)
        sampling_params = seq_group_metadata.sampling_params
        if sampling_params and sampling_params.prompt_logprobs is not None:
            inter_data.lora_prompt_mapping.append([lora_id] * query_len)
        elif not self.chunked_prefill_enabled or seq_group_metadata.do_sample:
            inter_data.lora_prompt_mapping.append([lora_id])
        else:
            inter_data.lora_prompt_mapping.append([])

    def _compute_multi_modal_input(self, inter_data: InterDataForSeqGroup,
                                   seq_group_metadata: SequenceGroupMetadata):
        """If multi-modal data is given, add it to the input."""
        # NOTE: mm_kwargs only includes the subset of multi-modal items that
        # intersect with the current prefill positions.
        positions = inter_data.input_positions[0]
        mm_kwargs, placeholder_maps = MultiModalPlaceholderMap.from_seq_group(
            seq_group_metadata,
            range(positions[0], positions[0] + len(positions)))

        # M-RoPE requires mrope_positions even for plain text; return early
        # when mm_kwargs is empty only if inter_data.is_prompt is False.
        if not mm_kwargs and not inter_data.is_prompt:
            return

        inter_data.multi_modal_kwargs = mm_kwargs
        inter_data.multi_modal_placeholder_maps = placeholder_maps

        # special processing for mrope position deltas.
        if self.runner.model_config.uses_mrope:
            image_grid_thw = mm_kwargs.get("image_grid_thw", None)
            video_grid_thw = mm_kwargs.get("video_grid_thw", None)
            audio_feature_lengths = mm_kwargs.get("audio_feature_lengths",
                                                  None)

            second_per_grid_ts = mm_kwargs.get("second_per_grid_ts", None)
            use_audio_in_video = mm_kwargs.get("use_audio_in_video", False)
            hf_config = self.runner.model_config.hf_config

            inter_data.mrope_input_positions = [None] * inter_data.n_seqs
            for seq_idx in range(inter_data.n_seqs):
                seq_data = seq_group_metadata.seq_data[
                    inter_data.seq_ids[seq_idx]]
                token_ids = seq_data.get_token_ids()

                mrope_input_positions, mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        context_len=inter_data.context_lens[seq_idx],
                        seq_len=inter_data.seq_lens[seq_idx],
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        encoder_seq_len = 0

        if self.runner.model_config.is_encoder_decoder:
            encoder_seq_len = seq_group_metadata.encoder_seq_data.get_len()

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True,
            encoder_seq_len=encoder_seq_len)

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)
        for per_seq_group_fn in self.per_seq_group_compute_fns:
            per_seq_group_fn(inter_data, seq_group_metadata)

    def _use_captured_graph(self,
                            batch_size: int,
                            decode_only: bool,
                            max_decode_seq_len: int,
                            max_encoder_seq_len: int = 0) -> bool:
        return (decode_only and not self.runner.model_config.enforce_eager
                and max_decode_seq_len <= self.runner.max_seq_len_to_capture
                and max_encoder_seq_len <= self.runner.max_seq_len_to_capture
                and batch_size <= self.runner.max_batchsize_to_capture)

    def _get_cuda_graph_pad_size(self,
                                 num_seqs: int,
                                 max_decode_seq_len: int,
                                 max_encoder_seq_len: int = 0) -> int:
        """
        Determine the number of padding sequences required for running in
        CUDA graph mode. Returns -1 if CUDA graphs cannot be used.

        In the multi-step + chunked-prefill case, only the first step
        has Prefills (if any). The rest of the steps are guaranteed to be all
        decodes. In this case, we set up the padding as if all the sequences
        are decodes so we may run all steps except the first step in CUDA graph
        mode. The padding is accounted for in the multi-step `advance_step`
        family of functions.

        Args:
            num_seqs (int): Number of sequences scheduled to run.
            max_decode_seq_len (int): Greatest of all the decode sequence
                lengths. Used only in checking the viablility of using
                CUDA graphs.
            max_encoder_seq_len (int, optional): Greatest of all the encode
                sequence lengths. Defaults to 0. Used only in checking the
                viability of using CUDA graphs.
        Returns:
            int: Returns the determined number of padding sequences. If
                CUDA graphs is not viable, returns -1.
        """
        is_mscp: bool = self.runner.scheduler_config.is_multi_step and \
                    self.runner.scheduler_config.chunked_prefill_enabled
        decode_only = self.decode_only or is_mscp
        if not decode_only:
            # Early exit so we can treat num_seqs as the batch_size below.
            return -1

        # batch_size out of this function refers to the number of input
        # tokens being scheduled. This conflation of num_seqs as batch_size
        # is valid as this is a decode-only case.
        batch_size = num_seqs
        if not self._use_captured_graph(batch_size, decode_only,
                                        max_decode_seq_len,
                                        max_encoder_seq_len):
            return -1

        graph_batch_size = self.runner.vllm_config.pad_for_cudagraph(
            batch_size)
        assert graph_batch_size >= batch_size
        return graph_batch_size - batch_size

    def build(self) -> ModelInputForGPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = list[int]()
        inputs_embeds_list = list[torch.Tensor]()
        token_types = list[int]()
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)
            for cur_token_types in inter_data.token_types:
                token_types.extend(cur_token_types)
            if inter_data.inputs_embeds is not None:
                inputs_embeds_list.append(
                    inter_data.inputs_embeds.to(
                        dtype=self.runner.model_config.dtype,
                        device=self.runner.device))
        inputs_embeds: Optional[torch.Tensor]
        if len(inputs_embeds_list) == 0:
            inputs_embeds = None
        else:
            inputs_embeds = torch.cat(inputs_embeds_list, dim=0).to(
                dtype=self.runner.model_config.dtype,
                device=self.runner.device)
            assert len(inputs_embeds) == len(input_tokens)

        if not input_tokens and inputs_embeds is None:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        mrope_input_positions: Optional[List[List[int]]] = None
        if any(inter_data.mrope_input_positions is not None
               for inter_data in self.inter_data_list):
            mrope_input_positions = [[] for _ in range(3)]
            for idx in range(3):
                for inter_data in self.inter_data_list:
                    msections = inter_data.mrope_input_positions
                    if msections is None:
                        for _seq_input_positions in inter_data.input_positions:
                            mrope_input_positions[idx].extend(
                                _seq_input_positions)
                    else:
                        for _seq_mrope_input_positions in msections:
                            mrope_input_positions[idx].extend(
                                _seq_mrope_input_positions[idx])
            input_positions = None
        else:
            input_positions = []
            for inter_data in self.inter_data_list:
                for cur_input_positions in inter_data.input_positions:
                    input_positions.extend(cur_input_positions)

        seq_lens = []
        query_lens = []
        max_decode_seq_len = 0
        max_encoder_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            query_lens.extend(inter_data.query_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
                if self.runner.model_config.is_encoder_decoder:
                    max_encoder_seq_len = max(max_encoder_seq_len,
                                              inter_data.encoder_seq_len)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        cuda_graph_pad_size = self._get_cuda_graph_pad_size(
            num_seqs=len(seq_lens),
            max_decode_seq_len=max_decode_seq_len,
            max_encoder_seq_len=max_encoder_seq_len)

        batch_size = len(input_tokens)
        if cuda_graph_pad_size != -1:
            # If cuda graph can be used, pad tensors accordingly.
            # See `capture_model` API for more details.
            # vLLM uses cuda graph only for decoding requests.
            batch_size += cuda_graph_pad_size

        # Tokens and positions.
        if cuda_graph_pad_size:
            input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory)

        token_types_tensor = async_tensor_h2d(token_types, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory) \
                                                if token_types else None

        if mrope_input_positions is not None:
            for idx in range(3):
                mrope_input_positions[idx].extend(
                    itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(mrope_input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        else:
            input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        # Sequence and query lengths.
        if cuda_graph_pad_size:
            seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            lora_requests = set(r for data in self.inter_data_list
                                for r in data.lora_requests)
            lora_index_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_index_mapping)
                for inter_data in self.inter_data_list
            ])
            if cuda_graph_pad_size:
                lora_index_mapping.extend(
                    itertools.repeat(0, cuda_graph_pad_size))
            lora_prompt_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_prompt_mapping)
                for inter_data in self.inter_data_list
            ])

            lora_mapping = LoRAMapping(
                **dict(index_mapping=lora_index_mapping,
                       prompt_mapping=lora_prompt_mapping,
                       is_prefill=not self.decode_only))

        # Multi-modal data.
        multi_modal_kwargs_list = [
            data.multi_modal_kwargs for data in self.inter_data_list
            if data.multi_modal_kwargs is not None
        ]
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            inputs_embeds=inputs_embeds,
            input_positions=input_positions_tensor,
            token_types=token_types_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids)


class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForGPU]
    _builder_cls: Type[ModelInputForGPUBuilder]
    builder: ModelInputForGPUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.max_batchsize_to_capture = \
            self.vllm_config.compilation_config.max_capture_size

        #
        self.graph_runners: List[Dict[Tuple[int, bool], CUDAGraphRunner]] = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.

        self.has_inner_state = model_config.has_inner_state

        self.in_profile_run = False

        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max seq len to capture / block size).
        self.graph_block_tables = np.zeros(
            (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
            dtype=np.int32)

        self.cross_layer_shared_graph_block_tables = np.zeros(
            (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
            dtype=np.int32)

        # Attention-free but stateful models like Mamba need a placeholder attn
        # backend, as the attention metadata is needed to manage internal state.
        # However we must bypass attention selection altogether for some models
        # used for speculative decoding to avoid a divide-by-zero in
        # model_config.get_head_size()
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        ) if needs_attn_backend else None
        if self.attn_backend:
            self.attn_state = self.attn_backend.get_state_cls()(
                weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.sampler = get_sampler()

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceGroupToSample object. In Pipeline-Parallel, we have
        # more than 1 Scheduler, resulting in a potential back-to-back
        # prepare_model_inputs() call. This clobbers the cached
        # SequenceGroupToSample objects, as we reset the cache during
        # every prepare_model_inputs() call.
        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler(self.device) as m:
            time_before_load = time.perf_counter()
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                assert supports_lora(
                    self.model
                ), f"{self.model.__class__.__name__} does not support LoRA yet."

                if supports_multimodal(self.model):
                    logger.warning(
                        "Regarding multimodal models, vLLM currently "
                        "only supports adding LoRA to language model.")

                # Use get_text_config() in case of multimodal models
                text_config = self.model_config.hf_config.get_text_config()

                self.lora_manager = LRUCacheWorkerLoRAManager(
                    self.scheduler_config.max_num_seqs,
                    self.scheduler_config.max_num_batched_tokens,
                    self.vocab_size,
                    self.lora_config,
                    self.device,
                    self.model.embedding_modules,
                    self.model.embedding_padding_modules,
                    max_position_embeddings=text_config.
                    max_position_embeddings,
                )
                self.model = self.lora_manager.create_lora_manager(self.model)
            time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)


        if self.vllm_config.compilation_config.level ==\
            CompilationLevel.DYNAMO_AS_IS and supports_dynamo():
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config)
            compilation_counter.dynamo_as_is_count += 1
            self.model = torch.compile(
                self.model,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)

    def get_model(self) -> nn.Module:
        return self.model

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForGPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        self.builder.prepare(finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            try:
                self.builder.add_seq_group(seq_group_metadata)
            except Exception as e:
                # Raise an exception that tracks the ID of the bad request
                raise InputProcessingError(seq_group_metadata.request_id,
                                           str(e)) from e

        self.builder.reset_cached_inter_data()

        return self.builder.build()  # type: ignore

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        self._dummy_run(max_num_batched_tokens, max_num_seqs)

    def _add_dummy_loras(self, num_loras: int) -> list[LoRARequest]:
        assert num_loras > 0
        assert self.lora_manager is not None

        dummy_lora_requests: list[LoRARequest] = []
        with self.lora_manager.dummy_lora_cache():
            for idx in range(num_loras):
                lora_id = idx + 1
                dummy_lora_request = LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                 rank=LORA_WARMUP_RANK)
                dummy_lora_requests.append(dummy_lora_request)
        return dummy_lora_requests

    def _remove_dummy_loras(self):
        # Remove dummy loras.
        assert self.lora_manager is not None
        self.remove_all_loras()

    def _dummy_run(self,
                   max_num_batched_tokens: int,
                   max_num_seqs: int = 1) -> None:
        with self.set_in_profile_run():
            # Enable top-k sampling to reflect the accurate memory usage.
            sampling_params = \
                SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

            # This represents the maximum number of different requests
            # that will have unique loras, and therefore the max amount of
            # memory consumption. Create dummy lora request copies from the
            # lora request passed in, which contains a lora from the lora
            # warmup path.
            dummy_lora_requests: List[LoRARequest] = []
            dummy_lora_requests_per_seq: List[LoRARequest] = []
            if self.lora_config:
                dummy_lora_requests = self._add_dummy_loras(
                    self.lora_config.max_loras)
                assert len(dummy_lora_requests) == self.lora_config.max_loras
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

            # Profile memory usage with max_num_sequences sequences and the
            # total number of tokens equal to max_num_batched_tokens.
            seqs: List[SequenceGroupMetadata] = []
            # Additional GPU memory may be needed for multi-modal encoding,
            # which needs to be accounted for when calculating the GPU blocks
            # for vLLM blocker manager.
            # To exercise the worst scenario for GPU memory consumption,
            # the number of seqs (batch_size) is chosen to maximize the number
            # of images processed.

            max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
                self.model_config)
            if max_mm_tokens > 0:
                max_num_seqs_orig = max_num_seqs
                max_num_seqs = min(max_num_seqs,
                                   max_num_batched_tokens // max_mm_tokens)
                if max_num_seqs < 1:
                    expr = (f"min({max_num_seqs_orig}, "
                            f"{max_num_batched_tokens} // {max_mm_tokens})")
                    logger.warning(
                        "Computed max_num_seqs (%s) to be less than 1. "
                        "Setting it to the minimum value of 1.", expr)
                    max_num_seqs = 1

            batch_size = 0
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                batch_size += seq_len

                dummy_data = self.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                              seq_len,
                                              self.mm_registry)

                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    lora_request=dummy_lora_requests_per_seq[group_id]
                    if dummy_lora_requests_per_seq else None,
                    multi_modal_data=dummy_data.multi_modal_data,
                    multi_modal_placeholders=dummy_data.
                    multi_modal_placeholders,
                )
                seqs.append(seq)

            # Run the model with the dummy inputs.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            # use an empty tensor instead of `None`` to force Dynamo to pass
            # it by reference, rather by specializing on the value ``None``.
            # the `dtype` argument does not matter, and we use `float32` as
            # a placeholder (it has wide hardware support).
            # it is important to create tensors inside the loop, rather than
            # multiplying the list, to avoid Dynamo from treating them as
            # tensor aliasing.
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            finished_requests_ids = [seq.request_id for seq in seqs]
            model_input = self.prepare_model_input(
                seqs, finished_requests_ids=finished_requests_ids)
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = \
                    self.model.make_empty_intermediate_tensors(
                    batch_size=batch_size,
                    dtype=self.model_config.dtype,
                    device=self.device)

            # Disable KV Scale Calculation for dummy data during profile run
            if model_input.attn_metadata is not None:
                model_input.attn_metadata.enable_kv_scales_calculation = False

            self.execute_model(model_input, kv_caches, intermediate_tensors)
            torch.cuda.synchronize()
            if self.lora_config:
                self._remove_dummy_loras()

            return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing cudagraphs for decoding. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI. "
                    "If out-of-memory error occurs during cudagraph capture,"
                    " consider decreasing `gpu_memory_utilization` or "
                    "switching to eager mode. You can also reduce the "
                    "`max_num_seqs` as needed to decrease memory usage.")
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros(max_batch_size,
                                   dtype=torch.long,
                                   device=self.device)
        input_positions = torch.zeros(max_batch_size,
                                      dtype=torch.long,
                                      device=self.device)
        inputs_embeds = torch.zeros(
            (max_batch_size, self.model_config.get_hidden_size()),
            dtype=self.model_config.dtype,
            device=self.device)
        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions,
                                         (3, 1)).cuda(device=self.device)
        # Prepare dummy previous_hidden_states only if needed by the model.
        # This is used by draft models such as EAGLE.
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(
                self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [max_batch_size,
                 self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device)

        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        dummy_lora_id: Optional[int] = None
        dummy_lora_request: LoRARequest = []
        if self.lora_config:
            # The goal is to capture the LoRA kernels in cuda graphs.
            # for this purpose, as single dummy lora is sufficient.
            dummy_lora_requests = self._add_dummy_loras(num_loras=1)
            assert len(dummy_lora_requests) == 1
            dummy_lora_request = dummy_lora_requests[0]
            dummy_lora_id = dummy_lora_request.lora_int_id

        with self.attn_state.graph_capture(max_batch_size), graph_capture(
                self.device) as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for virtual_engine in range(
                    self.parallel_config.pipeline_parallel_size):
                # We need to not only iterate over batch sizes, but also whether
                # to use inputs_embeds or not, hence we use the cartesian
                # product.
                cudagraph_capture_sizes = self.vllm_config.compilation_config\
                    .cudagraph_capture_sizes
                cudagraph_inputs_embeds = ((
                    True, False) if self.model_config.enable_prompt_embeds else
                                           (False, ))
                compilation_cases = itertools.product(
                    cudagraph_capture_sizes,
                    cudagraph_inputs_embeds,
                )
                # Only rank 0 should print progress bar during capture
                if get_tensor_model_parallel_rank() == 0:
                    compilation_cases = tqdm(
                        list(compilation_cases),
                        disable=not self.load_config.use_tqdm_on_load,
                        desc="Capturing CUDA graph shapes")
                for batch_size, use_inputs_embeds in compilation_cases:
                    attn_metadata = (
                        self.attn_state.graph_capture_get_metadata_for_batch(
                            batch_size,
                            is_encoder_decoder_model=self.model_config.
                            is_encoder_decoder))
                    # Disable KV Scale Calculation for graph capture
                    attn_metadata.enable_kv_scales_calculation = False
                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(index_mapping=[dummy_lora_id] * batch_size,
                                   prompt_mapping=[dummy_lora_id] * batch_size,
                                   is_prefill=False))
                        self.set_active_loras(set([dummy_lora_request]),
                                              lora_mapping)

                    graph_runner = CUDAGraphRunner(
                        self.model, self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder)

                    capture_inputs = {
                        "input_ids":
                        input_tokens[:batch_size],
                        "inputs_embeds":
                        inputs_embeds[:batch_size]
                        if use_inputs_embeds else None,
                        "positions":
                        input_positions[..., :batch_size],
                        "intermediate_inputs":
                        intermediate_inputs[:batch_size]
                        if intermediate_inputs is not None else None,
                        "kv_caches":
                        kv_caches[virtual_engine],
                        "attn_metadata":
                        attn_metadata,
                        "memory_pool":
                        self.graph_memory_pool,
                        "stream":
                        graph_capture_context.stream
                    }
                    if previous_hidden_states is not None:
                        capture_inputs[
                            "previous_hidden_states"] = previous_hidden_states[:
                                                                               batch_size]

                    if self.has_inner_state:
                        # Only used by Mamba-based models CUDA graph atm (Jamba)
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(
                                batch_size)
                        })
                    if self.model_config.is_encoder_decoder:
                        # add the additional inputs to capture for
                        # encoder-decoder models.
                        self._update_inputs_to_capture_for_enc_dec_model(
                            capture_inputs)

                    with set_forward_context(attn_metadata, self.vllm_config,
                                             virtual_engine):
                        graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][(
                        batch_size, use_inputs_embeds)] = graph_runner

        if self.lora_config:
            self._remove_dummy_loras()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / GiB_bytes)

    def _update_inputs_to_capture_for_enc_dec_model(self,
                                                    capture_inputs: Dict[str,
                                                                         Any]):
        """
        Updates the set of input tensors needed for CUDA graph capture in an
        encoder-decoder model.

        This method modifies the provided `capture_inputs` dictionary by
        adding tensors specific to encoder-decoder specific models that
        need to be captured for CUDA Graph replay.
        """
        # During the decode phase encoder_input_ids and encoder_positions are
        # unset. Do the same thing for graph capture.
        capture_inputs["encoder_input_ids"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)
        capture_inputs["encoder_positions"] = torch.tensor([],
                                                           dtype=torch.long,
                                                           device=self.device)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input = \
            ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, model_input.seq_lens,
                model_input.query_lens, self.device, self.pin_memory,
                generators, self.sampling_metadata_cache)
        else:
            sampling_metadata = None
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            use_inputs_embeds = model_input.inputs_embeds is not None
            model_executable = self.graph_runners[virtual_engine][(
                graph_batch_size, use_inputs_embeds)]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    inputs_embeds=model_input.inputs_embeds,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(
                        multi_modal_kwargs,
                        device=self.device,
                    ),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if self.is_driver_worker:
            if model_input.async_callback is not None:
                model_input.async_callback()

            # Sample the next token.
            assert isinstance(self.sampler, Sampler)
            orig_include_gpu_probs = self.sampler.include_gpu_probs_tensor
            if model_input.inputs_embeds is not None:
                self.sampler.include_gpu_probs_tensor = True

            output: SamplerOutput = self.sampler(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            if (self.observability_config is not None
                    and self.observability_config.collect_model_forward_time
                    and output is not None):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                # If there are multiple workers, we are still tracking the
                # latency from the start time of the driver worker to the end
                # time of the driver worker. The model forward time will then
                # end up covering the communication time as well.
                output.model_forward_time = (orig_model_forward_time +
                                             model_forward_time)

        if model_input.inputs_embeds is not None:
            if self.is_driver_worker:
                sampled_token_ids = []
                valid_outputs = []
                for sequence_group_output in output.outputs:
                    if len(sequence_group_output.samples) == 0:
                        continue
                    assert len(sequence_group_output.samples) == 1
                    valid_outputs.append(sequence_group_output)
                    sampled_token_ids.append(
                        sequence_group_output.samples[0].output_token)
                sampled_token_ids = torch.tensor(sampled_token_ids).to(
                    self.device)
                sampled_token_ids = broadcast_tensor_dict(
                    {"sampled_token_ids":
                     sampled_token_ids})["sampled_token_ids"]
            else:
                sampled_token_ids = broadcast_tensor_dict(
                )["sampled_token_ids"]
            if len(sampled_token_ids) > 0:
                sampled_token_embeds = \
                    self.model.get_input_embeddings(sampled_token_ids)
                if self.is_driver_worker:
                    self.sampler.include_gpu_probs_tensor = \
                        orig_include_gpu_probs
                    for i, sequence_group_output in enumerate(valid_outputs):
                        sequence_group_output.samples[0].output_embed = \
                            sampled_token_embeds[i]

        if not self.is_driver_worker:
            return []

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]

    def need_recv_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to receive kv-cache from the other worker.
        We need to receive KV when
            1. current vLLM instance is KV cache consumer/decode vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run

        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = (kv_caches[0].numel() == 0)
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return self.vllm_config.kv_transfer_config.is_kv_consumer and (
            not is_profile_run) and is_prefill_run

    def need_send_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to send kv-cache to the other worker.
        We need to send KV when
            1. current vLLM instance is KV cache producer/prefill vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run

        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = (kv_caches[0].numel() == 0)
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return self.vllm_config.kv_transfer_config.is_kv_producer and (
            not is_profile_run) and is_prefill_run


# NOTE: this is nn.Module so the profiler can properly capture/group
#  kernels calls made within the graph
class CUDAGraphRunner(nn.Module):

    def __init__(self, model: nn.Module, backend_name: str,
                 attn_state: AttentionState, is_encoder_decoder_model: bool):
        super().__init__()
        self.model = model
        self.backend_name = backend_name
        self.attn_state = attn_state

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._is_encoder_decoder_model = is_encoder_decoder_model

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_inputs: Optional[IntermediateTensors],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ):
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.compile
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                positions=positions,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )
        # Wait for the warm up operations to finish before proceeding with
        # Graph Capture.
        torch.cuda.synchronize()
        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_or_intermediate_states = self.model(
                input_ids=input_ids,
                **({
                    "inputs_embeds": inputs_embeds,
                } if inputs_embeds is not None else {}),
                positions=positions,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )

            if isinstance(output_hidden_or_intermediate_states, torch.Tensor):
                hidden_or_intermediate_states = weak_ref_tensor(
                    output_hidden_or_intermediate_states)
            elif isinstance(output_hidden_or_intermediate_states,
                            IntermediateTensors):
                hidden_or_intermediate_states = IntermediateTensors(
                    tensors={
                        key: weak_ref_tensor(value)
                        for key, value in
                        output_hidden_or_intermediate_states.tensors.items()
                    })

            del output_hidden_or_intermediate_states
            # make sure `output_hidden_or_intermediate_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids":
            input_ids,
            **({
                "inputs_embeds": inputs_embeds,
            } if inputs_embeds is not None else {}),
            "positions":
            positions,
            "kv_caches":
            kv_caches,
            **self.attn_state.get_graph_input_buffers(
                attn_metadata, self._is_encoder_decoder_model),
            **kwargs,
        }
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.tensors)
        if get_pp_group().is_last_rank:
            self.output_buffers = {
                "hidden_states": hidden_or_intermediate_states
            }
        else:
            self.output_buffers = hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        if positions is not None:
            # in some case like MLA, it will reuse positions in metadata
            # but truncate them to the original size
            # so the shape is not padded, we need to copy partial only
            self.input_buffers["positions"][:positions.shape[0]].copy_(
                positions, non_blocking=True)
        if inputs_embeds is not None:
            self.input_buffers["inputs_embeds"][:inputs_embeds.shape[0]].copy_(
                inputs_embeds, non_blocking=True)

        if self.backend_name != "NO_ATTENTION":
            self.input_buffers["slot_mapping"].copy_(
                attn_metadata.slot_mapping, non_blocking=True)

        self.attn_state.prepare_graph_input_buffers(
            self.input_buffers, attn_metadata, self._is_encoder_decoder_model)

        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)

        if "previous_hidden_states" in self.input_buffers:
            self.input_buffers["previous_hidden_states"].copy_(
                kwargs["previous_hidden_states"], non_blocking=True)

        if intermediate_tensors is not None:
            for key in intermediate_tensors.tensors:
                if key != "model_execute_time" and key != "model_forward_time":
                    self.input_buffers[key].copy_(intermediate_tensors[key],
                                                  non_blocking=True)
        if self._is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"].copy_(
                kwargs['encoder_input_ids'], non_blocking=True)
            self.input_buffers["encoder_positions"].copy_(
                kwargs['encoder_positions'], non_blocking=True)

        # Run the graph.
        self.graph.replay()
        # Return the output tensor.
        if get_pp_group().is_last_rank:
            return self.output_buffers["hidden_states"]

        return self.output_buffers
