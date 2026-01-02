# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install -r requirements.txt

COPY . .

EXPOSE 50051

CMD ["python", "services/grpc/server.py", "--hf-model", "microsoft/phi-3-mini-4k-instruct", "--device", "cuda", "--trust-remote-code"]
