MODEL_PATH="/path/to/parrot-audio"
SERVED_MODEL_NAME="parrot-audio"

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export VLLM_USE_TRANSFORMERS_AUDIO_ENCODER=0
export VLLM_COMPILE_AUDIO_TOWER=1
export VLLM_USE_TRANSFORMERS_MULTI_MODAL_PROJECTOR=0
export VLLM_COMPILE_MULTI_MODAL_PROJECTOR=1

vllm serve $MODEL_PATH --dtype auto --load-format mixed_precision --max-model-len 8192 --limit-mm-per-prompt "{\"audio\": 5, \"video\": 0, \"image\": 0}" --gpu-memory-utilization 0.8 -tp 2 --served-model-name $SERVED_MODEL_NAME --host 0.0.0.0 --port 8000