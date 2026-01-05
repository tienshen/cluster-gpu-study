# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorrt_llm/release:v0.7.1

WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    tritonclient[all] \
    pynvml \
    numpy

# Copy tools
COPY tools/ /workspace/tools/

# This is a builder image - use for creating TRT-LLM engines
# Then mount the output to Triton server container
