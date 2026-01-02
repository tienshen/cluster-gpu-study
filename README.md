# Cluster GPU Study

## Phase 1: Single GPU, production-shaped service

Goal: a runnable service + repeatable benchmark + credible observability.

Workflow:
1. Get it working locally first (fast iteration).
2. Containerize early-but-minimal (as soon as it runs end-to-end).
3. Keep every later benchmark the same environment, different hardware.

Add:
- Load generator.
- Metrics/traces.
- A baseline report (p50/p90/p99, TTFT/TPOT, GPU util, HBM, queueing).

Definition of "done":
- One command to run the service (docker).
- One command to run the benchmark.
- Results land in `results/` with a config snapshot.

Think: reproducible single-node inference product.
## Phi-3 Mini gRPC service (local export + serve)

Quick path to download/export Phi-3 Mini, generate gRPC stubs, and serve from disk.

1) Install deps (example):
```
pip install torch transformers grpcio grpcio-tools
```

2) Export the model from Hugging Face:
```
python scripts/export_phi3.py --trust-remote-code --output-dir models/phi-3-mini
```

3) Generate gRPC stubs:
```
python scripts/generate_grpc_stubs.py
```

4) Run the gRPC server:
```
python -m services.grpc.server --model-path models/phi-3-mini --trust-remote-code --device cuda --bfloat16 --do-sample
```

5) Call it:
```
python -m services.grpc.client --prompt "Write a limerick about GPUs." --do-sample
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

