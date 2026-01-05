# Cluster GPU Study

**Production-ready LLM inference using NVIDIA Triton + TensorRT-LLM**

This project demonstrates high-performance LLM serving with:
- **TensorRT-LLM**: Optimized inference engines (3-5× faster than PyTorch)
- **Triton Inference Server**: Enterprise-grade serving with dynamic batching
- **Observability**: Built-in metrics, GPU telemetry, and benchmarking tools

---

## Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Build TensorRT-LLM Engine
**Windows:**
```cmd
build_engine.bat --model microsoft/phi-3-mini-4k-instruct --output models/phi3/1 --dtype float16
```

**Linux/Mac or manual:**
```bash
python tools/build_engine.py --model microsoft/phi-3-mini-4k-instruct --output models/phi3/1 --dtype float16
```

### 3. Start Triton Server
```bash
docker run --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v %cd%/models:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### 4. Run Benchmark
**Windows:**
```cmd
benchmark.bat --model phi3 --requests 100 --concurrency 4 --report results/phi3_baseline.json
```

**Linux/Mac or manual:**
```bash
python tools/benchmark.py --model phi3 --requests 100 --concurrency 4 --report results/phi3_baseline.json
```

---

## Architecture

```
┌─────────────┐
│   Client    │
│  (HTTP/gRPC)│
└──────┬──────┘
       │
┌──────▼─────────────────────┐
│  Triton Inference Server   │
│  • Dynamic batching        │
│  • Model versioning        │
│  • Prometheus metrics      │
└──────┬─────────────────────┘
       │
┌──────▼─────────────────────┐
│  TensorRT-LLM Backend      │
│  • Compiled GPU kernels    │
│  • KV-cache optimization   │
│  • FP16/INT8 quantization  │
└────────────────────────────┘
```

---

## Project Structure

```
cluster-gpu-study/
├── tools/
│   ├── build_engine.py      # Build TRT-LLM engines
│   └── benchmark.py         # Load testing & metrics
├── docker/
│   ├── Dockerfile.builder   # TensorRT-LLM build environment
│   └── Dockerfile.triton    # Triton server image
├── configs/
│   └── model_config.yaml    # Model parameters
├── models/                  # Triton model repository
│   └── phi3/
│       ├── 1/               # Version 1
│       │   └── model.plan   # TRT engine
│       └── config.pbtxt     # Triton config
└── results/                 # Benchmark outputs
```

---

## Performance Expectations

| Metric | PyTorch Baseline | TensorRT-LLM |
|--------|------------------|--------------|
| TPOT (ms/token) | ~18-20 | ~5-7 |
| TTFT (ms) | 300-400 | 100-150 |
| Throughput (tok/s) | 50-60 | 150-200 |
| GPU Utilization | 60-70% | 85-95% |

*Results on NVIDIA A100 40GB, batch_size=8, fp16*

---

## Configuration

Edit `configs/model_config.yaml` to adjust:
- Model precision (fp16/int8)
- Batch sizes and queue delays
- Input/output length limits
- Generation parameters

---

## Backup Branch

Previous Phase 1 implementation (custom gRPC + PyTorch) preserved in:
```bash
git checkout backup/phase1-grpc-service
```

## Phase 2: Multi-GPU on one node

Goal: scaling story without network complexity.

What changes:
- Same container + same benchmark harness.
- Deployment changes to multi-GPU config (tensor parallel / pipeline parallel / multi-instance).

The work becomes bottleneck hunting:
- PCIe/NVLink utilization.
- NCCL overhead.
- HBM pressure / KV cache growth.
- CPU becoming the limiter (tokenization, scheduling, launch overhead).

Definition of "done":
- Scaling curves (1 -> 2 -> 4 GPUs).
- Clear explanation of where scaling breaks and why.
- At least one mitigation tested (e.g., batch policy, parallelism degree, pinned memory, affinity).

## Phase 3: Two nodes (real cluster)

Goal: show you can handle distributed reality.

What changes:
- Scheduling + network now matter.
- You will see new failure modes:
  - Cross-node NCCL sensitivity.
  - Jitter and tail latency.
  - Startup/initialization cost.
  - Node heterogeneity.
  - Interruptions (spot/preemptible).

