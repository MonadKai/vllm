# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import torch

from vllm.logger import init_logger
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       LogitsProcessor,
                                                       MoveDirectionality)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


logger = init_logger(__name__)


class AntiRepetitionLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that detects repetitive token patterns and increases EOS token probability
    to encourage the model to stop generating when repetition is detected.
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        
        # Track requests that need anti-repetition processing
        self.anti_repetition_reqs: dict[int, tuple[list[int], int, int]] = {}
        
        # Track requests that have been boosted and should be removed next step
        self.boosted_requests: set[int] = set()
        
        # GPU tensors for efficient processing
        self.request_indices = self._device_tensor([], torch.int32)
        self.eos_token_ids = self._device_tensor([], torch.int32)
        self.eos_boost_tensor = self._device_tensor([], torch.float32)

    def is_argmax_invariant(self) -> bool:
        """Anti-repetition can change the outcome of argmax by boosting EOS token."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int], output_tok_ids: list[int]
    ) -> Optional[tuple[list[int], int, int]]:
        """Check if request needs anti-repetition processing."""
        if params.extra_args is None:
            return None
        repetition_threshold = params.extra_args.get("repetition_threshold", 3)
        eos_token_id = params.extra_args.get("eos_token_id", None)
        
        if eos_token_id is None or repetition_threshold <= 1:
            return None
        
        return output_tok_ids, eos_token_id, repetition_threshold

    def _detect_repetition(self, token_ids: list[int], threshold: int) -> bool:
        """
        Detect if the token sequence has repetitive patterns.
        
        Args:
            token_ids: List of generated token IDs
            threshold: Minimum number of repetitions to consider as problematic
            
        Returns:
            True if repetition is detected, False otherwise
        """
        logger.info(f"token_ids: {token_ids}, threshold: {threshold}")
        # Check for single token repetition
        if self._check_single_token_repetition(token_ids, threshold):
            logger.info(f"single token repetition detected")
            return True
            
        # Check for n-gram repetition (2-gram, 3-gram, etc.)
        for n in range(2, min(5, len(token_ids) // threshold + 1)):
            if self._check_ngram_repetition(token_ids, n, threshold):
                logger.info(f"{n}-gram repetition detected")
                return True
        
        logger.info(f"no repetition detected")
        return False

    def _check_single_token_repetition(self, token_ids: list[int], threshold: int) -> bool:
        """Check if the same token repeats at the end."""
        if len(token_ids) < threshold:
            return False
            
        last_token = token_ids[-1]
        count = 1
        
        # Count consecutive occurrences from the end
        for i in range(len(token_ids) - 2, -1, -1):
            if token_ids[i] == last_token:
                count += 1
            else:
                break
        
        return count >= threshold

    def _check_ngram_repetition(self, token_ids: list[int], n: int, threshold: int) -> bool:
        """Check if n-gram patterns repeat at the end."""
        if len(token_ids) < n * threshold:
            return False
            
        # Get the last n tokens
        last_ngram = token_ids[-n:]
        count = 1
        
        # Check for repetitions of this n-gram
        for i in range(len(token_ids) - 2 * n, -1, -n):
            if token_ids[i:i+n] == last_ngram:
                count += 1
            else:
                break
                
        return count >= threshold

    def update_state(self, batch_update: Optional[BatchUpdate]):
        """Update the processor state based on batch changes."""
        needs_update = process_dict_updates(
            self.anti_repetition_reqs, batch_update, self.add_request)
        
        # remove boosted requests first
        if self.boosted_requests:
            for req in self.boosted_requests:
                self.anti_repetition_reqs.pop(req, None)
            self.boosted_requests.clear()
            needs_update = True
        
        if self.anti_repetition_reqs:
            # Check for requests that have developed repetition patterns
            to_boost = []
            
            for index, (output_tok_ids, eos_token_id, threshold) in self.anti_repetition_reqs.items():
                if self._detect_repetition(output_tok_ids, threshold):
                    # Repetition detected, mark for EOS boosting
                    to_boost.append((index, eos_token_id))
                    
            if to_boost:
                needs_update = True
                # Update GPU tensors for EOS boosting
                reqs = [req for req, _ in to_boost]
                eos_tokens = [eos for _, eos in to_boost]
                
                self.request_indices = self._device_tensor(reqs, torch.int32)
                self.eos_token_ids = self._device_tensor(eos_tokens, torch.int32)
                # Set EOS boost values
                self.eos_boost_tensor = self._device_tensor([float("inf")] * len(reqs), torch.float32)
                
                # mark these requests as boosted, remove next update_state
                for req, _ in to_boost:
                    self.boosted_requests.add(req)
                
        # Update tensors if needed
        if needs_update and not self.anti_repetition_reqs:
            # No more requests to track, reset tensors
            self.request_indices = self._device_tensor([], torch.int32)
            self.eos_token_ids = self._device_tensor([], torch.int32)
            self.eos_boost_tensor = self._device_tensor([], torch.float32)

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        """Create device tensor from data."""
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply anti-repetition logic to logits."""
        if self.anti_repetition_reqs and len(self.request_indices) > 0:
            # Boost EOS token probability for requests with detected repetition
            logits[self.request_indices, self.eos_token_ids] += self.eos_boost_tensor
            logger.info(f"Applied anti-repetition boost to {len(self.request_indices)} requests")
        return logits


def process_dict_updates(
    req_entries: dict[int, T], batch_update: Optional[BatchUpdate],
    new_state: Callable[[SamplingParams, list[int], list[int]], Optional[T]]
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids,
                               output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
