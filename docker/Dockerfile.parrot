FROM docker.m.daocloud.io/vllm/vllm-openai:v0.11.0

# add vllm tei plugin support
# COPY dist/vllm_tei_plugin-0.10.1.1.tar.gz /tmp/vllm_tei_plugin-0.10.1.1.tar.gz
# RUN pip install /tmp/vllm_tei_plugin-0.10.1.1.tar.gz --no-deps && rm -rf /tmp/vllm_tei_plugin-0.10.1.1.tar.gz

# add vllm kubernetes plugin support
# COPY dist/vllm_kubernetes_plugin-0.1.0.tar.gz /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz
# RUN pip install /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz --no-deps && rm -rf /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz

# add audio support
RUN pip install librosa==0.11.0

# add transformers parrot audio support
COPY dist/transformers-4.57.0.tar.gz /tmp/transformers-4.57.0.tar.gz
RUN pip uninstall transformers -y && pip install /tmp/transformers-4.57.0.tar.gz --no-deps && rm -rf /tmp/transformers-4.57.0.tar.gz

# add vllm mixed precision model loader
COPY vllm/model_executor/model_loader/mixed_precision_loader.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/mixed_precision_loader.py
COPY vllm/model_executor/model_loader/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/__init__.py

# add vllm multimodal processing patch
COPY vllm/multimodal/processing.py /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing.py

# add vllm parrot audio support
COPY dist/parrot_commons-0.1.0.tar.gz /tmp/parrot_commons-0.1.0.tar.gz
RUN pip install /tmp/parrot_commons-0.1.0.tar.gz --no-deps && rm -rf /tmp/parrot_commons-0.1.0.tar.gz

COPY vllm/model_executor/models/parrot_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot_audio.py
COPY vllm/model_executor/models/parrot2_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot2_audio.py
COPY vllm/model_executor/models/parrot2_audio_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot2_audio_moe.py
COPY vllm/model_executor/models/registry.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py

# add vllm mm encoder warmup support
COPY vllm/v1/worker/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
# COPY vllm/config/multimodal.py /usr/local/lib/python3.12/dist-packages/vllm/config/multimodal.py

ENTRYPOINT ["vllm", "serve"]