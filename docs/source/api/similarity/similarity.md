基于 vLLM 部署的文本相似度计算 API 服务，支持中英文文本的语义相似度计算。
# 相似度计算方式
def _cosine_similarity_0_1(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    #  [-1, 1] --> [0, 1]
    return (cos + 1) / 2

# 相似度说明
| 相似度区间 | 说明 |
|-----------|------|
| 0.9-1.0 | 语义高度相似 |
| 0.7-0.9 | 语义相似 |
| 0.5-0.7 | 语义相关 |
| 0.0-0.5 | 语义不相关 |

# 模型启动方式
通embedding模型一样
vllm serve --model MODEL_PATH --task embedding ...

# 请求方式
curl -X POST http://localhost:8090/v1/similarities \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your model name",
    "text_1": "hi",
    "text_2": "hello"
  }'

# 请求参数说明
| 参数 | 是否必选 | 类型 | 说明 |
|-----------|------|------|
| text_1 |  是 |  string | 文本数据，UTF-8编码 |
| text_2 |  是 |  string | 文本数据，UTF-8编码 |

# 请求输出格式
{
  "id": "sim-3fba56eba13b4d9dad0d4f6436b75612",
  "object": "score",
  "created": 1770204146,
  "model": "your model name",
  "data": [0.7560316040266855],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}