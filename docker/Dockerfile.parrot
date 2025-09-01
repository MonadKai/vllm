FROM docker.m.daocloud.io/vllm/vllm-openai:v0.9.0.1

# add vllm tei plugin support
COPY dist/vllm_tei_plugin-0.9.0.1.tar.gz /tmp/vllm_tei_plugin-0.9.0.1.tar.gz
RUN pip install /tmp/vllm_tei_plugin-0.9.0.1.tar.gz --no-deps && rm -rf /tmp/vllm_tei_plugin-0.9.0.1.tar.gz

# add vllm kubernetes plugin support
# COPY dist/vllm_kubernetes_plugin-0.1.0.tar.gz /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz
# RUN pip install /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz --no-deps && rm -rf /tmp/vllm_kubernetes_plugin-0.1.0.tar.gz

# add transformers parrot audio support
COPY dist/transformers-4.52.4.tar.gz /tmp/transformers-4.52.4.tar.gz
RUN pip uninstall transformers -y && pip install /tmp/transformers-4.52.4.tar.gz --no-deps && rm -rf /tmp/transformers-4.52.4.tar.gz

# fix vllm torchaudio issue
RUN pip install torchaudio==2.7.0 -f https://download.pytorch.org/whl/cu128 && pip install librosa==0.11.0

# add vllm mixed precision model loader
COPY vllm/model_executor/model_loader/mixed_precision_loader.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/mixed_precision_loader.py
COPY vllm/model_executor/model_loader/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/__init__.py
COPY vllm/config.py /usr/local/lib/python3.12/dist-packages/vllm/config.py

# v0 worker's model runner
COPY vllm/worker/utils.py /usr/local/lib/python3.12/dist-packages/vllm/worker/utils.py
COPY vllm/worker/cpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/cpu_model_runner.py
COPY vllm/worker/model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py
COPY vllm/worker/multi_step_neuron_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/multi_step_neuron_model_runner.py
COPY vllm/worker/multi_step_neuronx_distributed_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/multi_step_neuronx_distributed_model_runner.py
COPY vllm/worker/neuron_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/neuron_model_runner.py
COPY vllm/worker/xpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/worker/xpu_model_runner.py

# v1 worker's model runner
COPY vllm/v1/worker/utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py
COPY vllm/v1/worker/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
COPY vllm/v1/worker/tpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/tpu_model_runner.py

# add vllm parrot audio support
COPY dist/parrot_commons-0.1.0.tar.gz /tmp/parrot_commons-0.1.0.tar.gz
RUN pip install /tmp/parrot_commons-0.1.0.tar.gz --no-deps && rm -rf /tmp/parrot_commons-0.1.0.tar.gz

COPY vllm/model_executor/models/parrot_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot_audio.py
COPY vllm/model_executor/models/parrot2_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot2_audio.py
COPY vllm/model_executor/models/parrot2_audio_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/parrot2_audio_moe.py
COPY vllm/model_executor/models/registry.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py

ENTRYPOINT ["vllm", "serve"]