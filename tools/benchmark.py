#!/usr/bin/env python3
"""Benchmark Triton Inference Server for LLM workloads."""
import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "tritonclient not found. Install with: pip install tritonclient[all]"
    ) from exc

try:
    import pynvml
except ImportError:
    pynvml = None


def _percentile(values: List[float], pct: float) -> float:
    """Calculate percentile from sorted list."""
    if not values:
        return 0.0
    rank = (len(values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


class GPUMonitor:
    """Monitor GPU utilization during benchmark."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None

    def __enter__(self):
        if pynvml is None:
            return self
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if pynvml:
            pynvml.nvmlShutdown()

    def _poll(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append({
                    "timestamp": time.time(),
                    "gpu_util": float(util.gpu),
                    "mem_used_mb": float(mem.used) / (1024 * 1024),
                })
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> Optional[Dict]:
        if not self.samples:
            return None
        utils = [s["gpu_util"] for s in self.samples]
        mems = [s["mem_used_mb"] for s in self.samples]
        return {
            "avg_gpu_util": statistics.mean(utils),
            "max_gpu_util": max(utils),
            "avg_mem_mb": statistics.mean(mems),
            "max_mem_mb": max(mems),
        }


def load_prompts(prompt_file: Optional[Path], inline: Optional[str]) -> List[str]:
    """Load prompts from file or use defaults."""
    if prompt_file and prompt_file.exists():
        lines = [l.strip() for l in prompt_file.read_text().splitlines() if l.strip()]
        if lines:
            return lines
    if inline:
        return [inline]
    return [
        "Explain how GPU acceleration improves LLM inference.",
        "Write three creative ideas for benchmarking distributed systems.",
        "Describe the trade-offs between latency and throughput in serving.",
        "What is the difference between TTFT and TPOT in LLM metrics?",
    ]


def run_benchmark(args) -> Dict:
    """Run load test against Triton server."""
    prompts = load_prompts(args.prompt_file, args.prompt)
    prompt_cycle = cycle(prompts)

    thread_local = threading.local()

    def _get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = grpcclient.InferenceServerClient(
                url=args.url,
                verbose=False,
            )
        return thread_local.client

    def _run_single(request_id: int, prompt: str) -> Dict:
        client = _get_client()
        
        # Prepare inputs (adjust based on your model's input signature)
        inputs = []
        inputs.append(grpcclient.InferInput("INPUT_ID", [1], "BYTES"))
        inputs[0].set_data_from_numpy(np.array([prompt.encode('utf-8')], dtype=object))
        
        # Configure generation parameters
        inputs.append(grpcclient.InferInput("MAX_TOKENS", [1], "INT32"))
        inputs[1].set_data_from_numpy(np.array([args.max_tokens], dtype=np.int32))
        
        inputs.append(grpcclient.InferInput("TEMPERATURE", [1], "FP32"))
        inputs[2].set_data_from_numpy(np.array([args.temperature], dtype=np.float32))
        
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))
        
        wall_start = time.perf_counter()
        try:
            response = client.infer(
                model_name=args.model,
                inputs=inputs,
                outputs=outputs,
                request_id=str(request_id),
            )
            wall_end = time.perf_counter()
            
            # Extract response (adjust based on output format)
            output_data = response.as_numpy("OUTPUT")
            
            return {
                "ok": True,
                "request_id": request_id,
                "latency_ms": (wall_end - wall_start) * 1000.0,
            }
        except Exception as exc:
            wall_end = time.perf_counter()
            return {
                "ok": False,
                "request_id": request_id,
                "error": str(exc),
                "latency_ms": (wall_end - wall_start) * 1000.0,
            }

    successes = []
    failures = []

    with GPUMonitor(args.gpu_index, args.gpu_sample_interval) as monitor:
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_run_single, i, next(prompt_cycle))
                for i in range(args.requests)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result["ok"]:
                    successes.append(result)
                else:
                    failures.append(result)
        wall_end = time.perf_counter()
        gpu_stats = monitor.summary()

    total_wall_ms = (wall_end - wall_start) * 1000.0
    throughput = len(successes) / ((wall_end - wall_start) or 1e-6)

    latencies = sorted(r["latency_ms"] for r in successes) if successes else []

    summary = {
        "total_requests": args.requests,
        "successful": len(successes),
        "failed": len(failures),
        "wall_time_ms": total_wall_ms,
        "throughput_rps": throughput,
        "latency_ms": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50) if latencies else 0.0,
            "p95": _percentile(latencies, 95) if latencies else 0.0,
            "p99": _percentile(latencies, 99) if latencies else 0.0,
        },
        "gpu": gpu_stats,
        "failures": failures[:10],  # Keep first 10 errors for debugging
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton Inference Server for LLM workloads"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name in Triton repository",
    )
    parser.add_argument(
        "--url",
        default="localhost:8001",
        help="Triton gRPC endpoint (default: localhost:8001)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to use for all requests",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max output tokens (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index for monitoring (default: 0)",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.5,
        help="GPU sampling interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to save JSON report",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Benchmarking Triton model: {args.model}")
    print(f"Server: {args.url}")
    print(f"Requests: {args.requests} (concurrency: {args.concurrency})")
    print("=" * 60)

    summary = run_benchmark(args)

    print("\n=== Results ===")
    print(json.dumps(summary, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
