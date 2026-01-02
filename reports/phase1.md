# Phase 1 Report

_Date: January 1, 2026_

## Deliverables Completed
- **Instrumented inference service**: The gRPC response now surfaces latency, TTFT, and token throughput directly from the server, enabling precise client-side analytics.
- **Mini load generator**: `scripts/load_generator.py` drives configurable request volumes, tracks TTFT / TPOT / throughput, and samples GPU metrics via NVML.
- **Observability hooks**: Lightweight GPU polling captures average/max utilization and memory, exported alongside load test metrics for quick regression spotting.
- **Container baseline**: Added a CUDA-ready `Dockerfile` (with `.dockerignore`) so the service can be packaged and deployed under `nvidia-docker`.

## How to Reproduce Key Measurements
1. Start the gRPC server (locally or inside Docker):
   ```bash
   python services/grpc/server.py --model-path models/phi-3-mini --device cuda --trust-remote-code
   ```
2. Run a sample load test with GPU telemetry and write a JSON report:
   ```bash
   python scripts/load_generator.py --requests 32 --concurrency 4 --report results/loadgen_phase1.json
   ```
3. Inspect the aggregated stats (TTFT, TPOT, throughput, GPU averages) from the printed summary or the saved JSON file.

## Risks & Next Steps
- **Model size vs. container footprint**: The Docker image currently downloads the model at runtime; pinning a local artifact or layering ONNX/TRT exports will avoid cold-start penalties.
- **Streaming**: We still return a single gRPC response. Promoting server-streaming would expose per-token latencies to clients directly.
- **Automation**: Integrating the load generator with CI smoke tests (smaller prompts) would prevent regressions in TTFT/TPOT.

Overall, Phase 1 establishes a measurable baseline (end-to-end latency + GPU telemetry) and a portable runtime surface for future optimization experiments.
