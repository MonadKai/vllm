import logging
from typing import Union

from fastapi import Request
from vllm.entrypoints.openai.protocol import (
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.tei.protocol import EmbedRequest, TruncationDirection


__all__ = ["TeiServingEmbed"]

logger = logging.getLogger(__name__)


def _convert_to_tei_embed_response(
    response: EmbeddingResponse,
) -> list[list[float]]:
    response.data.sort(key=lambda x: x.index)
    embeddings = [x.embedding for x in response.data]
    return embeddings


class TeiServingEmbed(OpenAIServingEmbedding):
    request_id_prefix = "tei-embed"

    def _convert_to_openai_embedding_request(
        self,
        request: EmbedRequest,
    ) -> EmbeddingCompletionRequest:
        return EmbeddingCompletionRequest(
            model=self._get_model_name(None),
            input=request.inputs,
            encoding_format="float",
            dimensions=None,
            user=None,
            truncate_prompt_tokens=self.max_model_len
            if request.truncate
            else None,
        )

    async def embed(
        self, request: EmbedRequest, raw_request: Request
    ) -> Union[list[list[float]], ErrorResponse]:
        if request.truncation_direction == TruncationDirection.LEFT:
            logger.error("truncate_direction is not supported")
            return self.create_error_response(
                "truncate_direction is not supported"
            )

        request_openai = self._convert_to_openai_embedding_request(request)
        response_openai = await super().create_embedding(
            request_openai, raw_request
        )
        if isinstance(response_openai, ErrorResponse):
            return response_openai
        response_tei = _convert_to_tei_embed_response(response_openai)
        return response_tei