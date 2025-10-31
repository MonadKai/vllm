# e.g. Fengshenbang/Erlangshen-MegatronBert-1.3B-Chinese
MODEL_PATH="/path/to/MegatronBert"
MODEL_PATH="/opt/huggingface/bairong-inc/MegatronBert"
SERVED_MODEL_NAME="megatron-bert"

if [ ! -f $MODEL_PATH/modules.json ] || [ ! -f $MODEL_PATH/config_sentence_transformers.json ] || [ ! -f $MODEL_PATH/sentence_bert_config.json ]; then
    echo "modules.json or config_sentence_transformers.json or sentence_bert_config.json not found in $MODEL_PATH"
    exit 1
fi

if [ ! -d $MODEL_PATH/1_Pooling ] || [ ! -f $MODEL_PATH/1_Pooling/config.json ]; then
    echo "1_Pooling not found in $MODEL_PATH"
    exit 1
fi

# HINT: vllm v0.10.0 does not support MegatronBertModel in V1 engine
export VLLM_USE_V1=0
vllm serve $MODEL_PATH --hf-overrides "{\"architectures\": [\"MegatronBertModel\"]}" --dtype float32 --served-model-name $SERVED_MODEL_NAME


# curl -X POST -H "Content-Type: application/json" -d '{"input": "hi", "model": "megatron-bert"}'  http://127.0.0.1:8000/v1/embeddings