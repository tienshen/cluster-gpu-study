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
