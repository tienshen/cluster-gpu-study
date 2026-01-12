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

## Cloud Deployment

This project is designed for cloud GPU environments. Local Windows development is for editing code/configs only.

### AWS Deployment (Recommended)

**1. Launch GPU Instance**
```bash
# Use Deep Learning AMI with Docker + NVIDIA drivers pre-installed
# Instance type: g4dn.xlarge (T4 GPU, ~$0.50/hr) or g5.xlarge (A10G, ~$1.00/hr)
aws ec2 run-instances \
    --image-id ami-0c2b8ca1dad447f8a \
    --instance-type g4dn.xlarge \
    --key-name YOUR_KEY \
    --security-group-ids sg-xxxxx \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100}'
```

**2. SSH and Setup**
```bash
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/YOUR_USERNAME/cluster-gpu-study.git
cd cluster-gpu-study
git checkout triton-overhaul

# Verify GPU
nvidia-smi
```

**3. Build TensorRT-LLM Engine**
```bash
# Build using NVIDIA's TensorRT-LLM container
docker run --gpus all -v $(pwd):/workspace \
    nvcr.io/nvidia/tensorrt_llm/release:v0.7.1 \
    bash -c "cd /workspace && python tools/build_engine.py \
        --model microsoft/phi-3-mini-4k-instruct \
        --output models/phi3/1 \
        --dtype float16 \
        --max-batch-size 8"
```

**4. Start Triton Server**
```bash
docker run -d --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models \
    --name triton-server \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

**5. Run Benchmarks**
```bash
# Install client dependencies
pip install -r requirements.txt

# Benchmark with increasing concurrency
for c in 1 2 4 8 16; do
    python tools/benchmark.py \
        --model phi3 \
        --requests 100 \
        --concurrency $c \
        --report results/phi3_c${c}.json
done
```

**6. Analyze Results**
```bash
# Download results to local machine
scp -i your-key.pem ubuntu@<instance-ip>:~/cluster-gpu-study/results/*.json ./local-results/

# Teardown instance when done
aws ec2 terminate-instances --instance-ids i-xxxxx
```

---

### GCP Deployment

**1. Launch GPU VM**
```bash
gcloud compute instances create triton-benchmark \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

**2. Setup (same as AWS steps 2-5)**

**3. Cleanup**
```bash
gcloud compute instances delete triton-benchmark --zone=us-central1-a
```

---

### Azure Deployment

**1. Launch GPU VM**
```bash
az vm create \
    --resource-group YOUR_RG \
    --name triton-benchmark \
    --image "microsoft-dsvm:ubuntu-2004:ubuntu-2004:latest" \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys
```

**2. Setup (same as AWS steps 2-5)**

**3. Cleanup**
```bash
az vm delete --resource-group YOUR_RG --name triton-benchmark --yes
```

---

## Development Workflow

**Local (Windows):**
- Edit tools, configs, documentation
- Commit and push changes
- No need to run inference locally

**Cloud (Linux + GPU):**
- Pull latest code
- Build engines and run benchmarks
- Download results for analysis
- Terminate instance

---

## Cost Optimization

| Provider | Instance | GPU | Cost/hr | Best For |
|----------|----------|-----|---------|----------|
| AWS | g4dn.xlarge | T4 | $0.526 | Development/testing |
| AWS | g5.xlarge | A10G | $1.006 | Production benchmarks |
| GCP | n1-standard-4 + T4 | T4 | $0.35-0.45 | Cost-optimized |
| Azure | NC4as_T4_v3 | T4 | $0.526 | Azure ecosystem |

**Tips:**
- Use spot/preemptible instances for 60-80% discount
- Terminate after each benchmark run
- Automate with Infrastructure-as-Code (Terraform/Pulumi)

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

