# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id in num_scheduled_tokens:
            request = self.requests[req_id]
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            if request.is_prefill_chunk:
                # Prefill: add placeholders for the tokens we are processing
                # so num_output_tokens + num_output_placeholders is correct
                # in _make_cached_request_data (used for batch/position building).
                request.num_output_placeholders += num_scheduled_tokens.get(
                    req_id, 0
                )
            else:
                # Decode: the request will generate one token plus spec tokens.
                cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new draft/spec tokens.
                request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        *,
        num_tokens_scheduled: int = 0,
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request,
            new_token_ids,
            num_tokens_scheduled=num_tokens_scheduled,
        )

        # Update the number of output placeholders. Prefill: subtract the
        # tokens we processed this step. Decode: subtract generated tokens.
        if request.is_prefill_chunk:
            to_subtract = num_tokens_scheduled
        else:
            to_subtract = len(new_token_ids)
        request.num_output_placeholders = max(
            0, request.num_output_placeholders - to_subtract
        )

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
