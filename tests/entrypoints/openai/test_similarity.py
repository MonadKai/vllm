# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vllm.entrypoints.openai.serving_embedding import (
    _chunk_token_ids,
    _pool_chunk_embeddings,
)


def test_chunk_token_ids_uses_overlap():
    token_ids = list(range(10))

    chunks = _chunk_token_ids(token_ids, chunk_size=4, chunk_overlap=1)

    assert chunks == [
        list(range(0, 4)),
        list(range(3, 7)),
        list(range(6, 10)),
    ]


def test_pool_chunk_embeddings_uses_token_count_weights_and_normalizes():
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    token_counts = [3, 1]

    pooled = _pool_chunk_embeddings(embeddings, token_counts)

    expected = np.array([0.75, 0.25])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled, expected.tolist())
