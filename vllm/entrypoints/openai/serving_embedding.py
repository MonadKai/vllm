# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import time
from collections.abc import AsyncGenerator
from typing import Final, Literal, Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingCompletionRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse,
                                              SimilarityRequest,
                                              SimilarityResponse,
                                              UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import (EmbeddingOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput)
from vllm.utils import merge_async_iterators

logger = init_logger(__name__)


def _chunk_token_ids(
    token_ids: list[int],
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[list[int]] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def _pool_chunk_embeddings(
    embeddings: list[list[float]],
    token_counts: list[int],
) -> list[float]:
    if len(embeddings) != len(token_counts):
        raise ValueError("embeddings and token_counts must have the same size")
    if not embeddings:
        raise ValueError("embeddings must not be empty")

    weights = np.array(token_counts, dtype="float32")
    if np.any(weights <= 0):
        raise ValueError("token_counts must be positive")

    pooled = np.average(np.array(embeddings, dtype="float32"),
                        axis=0,
                        weights=weights)
    norm = np.linalg.norm(pooled)
    if norm == 0:
        return pooled.tolist()
    return (pooled / norm).tolist()


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Embedding API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format

        model_name = self._get_model_name(request.model)
        request_id = f"embd-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        truncate_prompt_tokens = None

        if request.truncate_prompt_tokens is not None:
            if request.truncate_prompt_tokens <= self.max_model_len:
                truncate_prompt_tokens = request.truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")

        pooling_params = request.to_pooling_params()

        try:
            pooling_params.verify(self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

            if isinstance(request, EmbeddingChatRequest):
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    # In embedding requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.input,
                     truncate_prompt_tokens=truncate_prompt_tokens,
                     add_special_tokens=request.add_special_tokens,
                 )
        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: list[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(list[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_embedding_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
                encoding_format,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def request_output_to_embedding_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
    ) -> EmbeddingResponse:
        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            embedding_res = EmbeddingRequestOutput.from_base(final_res)

            item = EmbeddingResponseData(
                index=idx,
                embedding=_get_embedding(embedding_res.outputs,
                                         encoding_format),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )

    async def create_similarity(
            self, request: SimilarityRequest,
            raw_request: Request) -> Union[SimilarityResponse, ErrorResponse]:
        if request.long_text_strategy == "mean_pooling":
            chunked_response = await self._try_create_chunked_similarity(
                request, raw_request)
            if chunked_response is not None:
                return chunked_response

        request_openai = self._convert_to_openai_embedding_request(request)
        if request.long_text_strategy == "truncate" and \
                request_openai.truncate_prompt_tokens is None:
            request_openai.truncate_prompt_tokens = self.max_model_len

        response_openai = await self.create_embedding(request_openai,
                                                      raw_request)
        if isinstance(response_openai, ErrorResponse):
            return response_openai
        similarity_response = self._convert_to_similarity_response(
            response_openai)
        similarity_score = self._cosine_similarity_0_1(
            similarity_response[0], similarity_response[1])
        return SimilarityResponse(data=[float(similarity_score)],
                                  model=request_openai.model,
                                  usage=response_openai.usage)

    async def _try_create_chunked_similarity(
            self, request: SimilarityRequest,
            raw_request: Request
    ) -> Optional[Union[SimilarityResponse, ErrorResponse]]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            lora_request, _ = self._maybe_get_adapters(request)
            tokenizer = await self.engine_client.get_tokenizer(lora_request)
            text_token_ids = [
                self._tokenize_similarity_text(tokenizer, request.text_1,
                                               request.add_special_tokens),
                self._tokenize_similarity_text(tokenizer, request.text_2,
                                               request.add_special_tokens),
            ]
            special_tokens = self._num_similarity_special_tokens(
                tokenizer, request.add_special_tokens)
            body_token_limit = self.max_model_len - special_tokens
            if body_token_limit <= 0:
                return self.create_error_response(
                    "max_model_len is too small for special tokens.")

            if all(len(token_ids) + special_tokens <= self.max_model_len
                   for token_ids in text_token_ids):
                return None

            chunk_size = request.chunk_size or min(480, body_token_limit)
            chunk_size = min(chunk_size, body_token_limit)
            chunk_overlap = min(request.chunk_overlap, chunk_size - 1)

            chunk_inputs: list[list[int]] = []
            chunk_token_counts: list[list[int]] = []
            for token_ids in text_token_ids:
                chunks = _chunk_token_ids(token_ids, chunk_size,
                                          chunk_overlap)
                if not chunks:
                    chunks = [[]]
                chunk_token_counts.append(
                    [max(len(chunk), 1) for chunk in chunks])
                chunk_inputs.extend(
                    self._add_similarity_special_tokens(
                        tokenizer, chunk, request.add_special_tokens)
                    for chunk in chunks)
        except (TypeError, ValueError) as e:
            return self.create_error_response(str(e))

        embedding_request = EmbeddingCompletionRequest(
            model=self._get_model_name(request.model),
            input=chunk_inputs,
            encoding_format="float",
            dimensions=request.dimensions,
            user=request.user,
            truncate_prompt_tokens=None,
            add_special_tokens=False,
            priority=request.priority,
        )
        response_openai = await self.create_embedding(embedding_request,
                                                      raw_request)
        if isinstance(response_openai, ErrorResponse):
            return response_openai

        embeddings = self._convert_to_similarity_response(response_openai)
        first_count = len(chunk_token_counts[0])
        pooled_1 = _pool_chunk_embeddings(embeddings[:first_count],
                                          chunk_token_counts[0])
        pooled_2 = _pool_chunk_embeddings(embeddings[first_count:],
                                          chunk_token_counts[1])
        similarity_score = self._cosine_similarity_0_1(pooled_1, pooled_2)
        return SimilarityResponse(data=[float(similarity_score)],
                                  model=embedding_request.model,
                                  usage=response_openai.usage)

    def _tokenize_similarity_text(self, tokenizer, text: str,
                                  add_special_tokens: bool) -> list[int]:
        if (self.model_config.encoder_config is not None
                and self.model_config.encoder_config.get(
                    "do_lower_case", False)):
            text = text.lower()
        return tokenizer(text, add_special_tokens=False).input_ids

    def _num_similarity_special_tokens(
            self,
            tokenizer,
            add_special_tokens: bool,
    ) -> int:
        if not add_special_tokens:
            return 0
        if hasattr(tokenizer, "num_special_tokens_to_add"):
            return tokenizer.num_special_tokens_to_add(pair=False)
        return len(tokenizer.build_inputs_with_special_tokens([]))

    def _add_similarity_special_tokens(
            self,
            tokenizer,
            token_ids: list[int],
            add_special_tokens: bool,
    ) -> list[int]:
        if not add_special_tokens:
            return token_ids
        return tokenizer.build_inputs_with_special_tokens(token_ids)

    def _convert_to_similarity_response(
        self,
        response_openai: EmbeddingResponse,
    ) -> list[list[float]]:
        response_openai.data.sort(key=lambda x: x.index)
        embeddings = [x.embedding for x in response_openai.data]
        return embeddings

    def _convert_to_openai_embedding_request(
        self,
        request: SimilarityRequest,
    ) -> EmbeddingCompletionRequest:
        return EmbeddingCompletionRequest(
            model=self._get_model_name(request.model),
            input=[request.text_1, request.text_2],
            encoding_format="float",
            dimensions=request.dimensions,
            user=request.user,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
            priority=request.priority,
        )

    def _cosine_similarity_0_1(self, vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #  [-1, 1] --> [0, 1]
        return (cos + 1) / 2
