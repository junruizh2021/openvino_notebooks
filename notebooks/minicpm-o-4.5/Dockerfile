# MiniCPM-o 4.5 OpenVINO 演示 - 基于 Ubuntu 24.04
# 构建前请将模型目录复制到构建上下文中，例如：
#   cp -r /home/nvme-data/AI-models/MiniCPM-o-4_5-OV ./MiniCPM-o-4_5-OV
#   docker build -t minicpm-o-45-demo -f notebooks/minicpm-o-4.5/Dockerfile .
# 或从模型所在目录构建（需同时能访问到 demo 代码）：
#   docker build -t minicpm-o-45-demo -f Dockerfile \
#     --build-arg MODEL_SOURCE=/home/nvme-data/AI-models/MiniCPM-o-4_5-OV \
#     .

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装 Python 3.12 及系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 安装 Python 依赖（与 minicpm_o_4_5_demo.py 中一致）
RUN pip install --no-cache-dir \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
    "transformers==4.51.0" "openvino>=2025.1" "openvino-tokenizers>=2025.1" \
    "nncf>=2.16" "Pillow" "librosa" "soundfile" "gradio==6.0.0" \
    "accelerate" "einops" "onnxruntime" "hyperpyyaml" \
    "minicpmo-utils>=1.0.5" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 拷贝演示程序及其依赖
COPY minicpm_o_4_5_demo.py minicpm_o_4_5_helper.py gradio_helper.py notebook_utils.py ./

# 拷贝 OpenVINO 模型目录（构建时需将 MiniCPM-o-4_5-OV 放入构建上下文）
# 若从仓库外构建且模型在宿主机其他路径，可先复制到当前目录再 build
ARG MODEL_DIR=MiniCPM-o-4_5-OV
COPY ${MODEL_DIR} /app/MiniCPM-o-4_5-OV

# 将 demo 中的模型路径改为容器内路径
RUN sed -i 's|model_id = .*|model_id = "/app/MiniCPM-o-4_5-OV"|' minicpm_o_4_5_demo.py \
    && sed -i 's|ov_model_path = Path(.*)|ov_model_path = Path("/app/MiniCPM-o-4_5-OV")|' minicpm_o_4_5_demo.py

ENV OV_MODEL_PATH=/app/MiniCPM-o-4_5-OV

EXPOSE 7860

CMD ["python", "minicpm_o_4_5_demo.py", "--demo", "gradio"]
