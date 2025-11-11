import asyncio
import logging
from typing import Optional, Union

from fastapi import Request
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.protocol import RerankRequest
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput

from vllm.entrypoints.tei.protocol import RerankRequest as TeiRerankRequest

__all__ = ["TeiServingRerank"]


logger = logging.getLogger(__name__)


class TeiServingRerank(ServingScores):
    def _convert_to_openai_rerank_request(
        self, request: TeiRerankRequest
    ) -> RerankRequest:
        return RerankRequest(
            model=None,
            query=request.query,
            documents=request.texts,
            top_n=0,
            truncate_prompt_tokens=self.max_model_len
            if request.truncate
            else None,
        )

    def request_output_to_ranks(
        self,
        final_res_batch: list[PoolingRequestOutput],
    ) -> list[dict]:
        ranks: list[dict] = []
        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)
            item = {
                "index": idx,
                "score": classify_res.outputs.score,
            }
            ranks.append(item)
        return ranks

    async def do_rerank(
        self, request: RerankRequest, raw_request: Optional[Request] = None
    ) -> Union[list[dict], ErrorResponse]:
        """
        Rerank API based on JinaAI's rerank API; implements the same
        API interface. Designed for compatibility with off-the-shelf
        tooling, since this is a common standard for reranking APIs
        See example client implementations at
        https://github.com/infiniflow/ragflow/blob/main/rag/llm/rerank_model.py
        numerous clients use this standard.
        """

        request_id = f"tei-rerank-{self._base_request_id(raw_request)}"
        documents = request.documents

        try:
            final_res_batch = await self._run_scoring(
                request.query,
                documents,
                request,
                request_id,
                raw_request,
            )
            if isinstance(final_res_batch, ErrorResponse):
                return final_res_batch

            return self.request_output_to_ranks(final_res_batch)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def rerank(
        self, request: TeiRerankRequest, raw_request: Request
    ) -> Union[list[dict], ErrorResponse]:
        if request.return_text:
            logger.exception("return_text is not supported")
            return self.create_error_response("return_text is not supported")

        if request.raw_scores:
            logger.exception("raw_scores is not supported")
            return self.create_error_response("raw_scores is not supported")

        rerank_request = self._convert_to_openai_rerank_request(request)
        response_tei = await self.do_rerank(rerank_request, raw_request)
        return response_tei