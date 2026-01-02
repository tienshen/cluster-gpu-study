#!/usr/bin/env python3
"""Minimal load generator for the gRPC text generation service."""
import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

import grpc

from services.grpc import text_generation_pb2
from services.grpc import text_generation_pb2_grpc

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    rank = (len(values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


class GPUMonitor:
    """Polls NVML for utilization stats while the load test is running."""

    def __init__(self, device_index: int, interval_s: float) -> None:
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
        if self._thread is not None:
            self._thread.join()
        if pynvml is not None:
            pynvml.nvmlShutdown()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                sample = {
                    "timestamp": time.time(),
                    "gpu_util": float(util.gpu),
                    "mem_used_mb": float(mem.used) / (1024 * 1024),
                    "mem_total_mb": float(mem.total) / (1024 * 1024),
                }
                self.samples.append(sample)
            except Exception:  # pragma: no cover - NVML failures are non-fatal
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> Optional[Dict[str, float]]:
        if not self.samples:
            return None
        gpu_utils = [s["gpu_util"] for s in self.samples]
        mem_used = [s["mem_used_mb"] for s in self.samples]
        mem_total = self.samples[-1]["mem_total_mb"]
        return {
            "avg_gpu_util": statistics.mean(gpu_utils),
            "max_gpu_util": max(gpu_utils),
            "avg_mem_used_mb": statistics.mean(mem_used),
            "max_mem_used_mb": max(mem_used),
            "mem_total_mb": mem_total,
        }


def _load_prompts(prompt_file: Optional[Path], inline_prompt: Optional[str]) -> List[str]:
    if prompt_file:
        lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines()]
        prompts = [line for line in lines if line]
        if prompts:
            return prompts
    if inline_prompt:
        return [inline_prompt]
    return [
        "Explain how GPU tensor cores accelerate matrix multiplication.",
        "Give me three creative GPU benchmarking ideas.",
        "Write a limerick about distributed inference.",
        "Describe why TTFT matters for chat UX.",
    ]


def run_load(args) -> Dict[str, object]:
    prompts = _load_prompts(args.prompt_file, args.prompt)
    prompts_cycle = cycle(prompts)

    thread_local = threading.local()

    def _get_stub():
        if not hasattr(thread_local, "stub"):
            channel = grpc.insecure_channel(f"{args.host}:{args.port}")
            thread_local.stub = text_generation_pb2_grpc.TextGenerationStub(channel)
        return thread_local.stub

    def _run_single(request_id: int, prompt_text: str) -> Dict[str, object]:
        stub = _get_stub()
        request = text_generation_pb2.GenerateRequest(
            prompt=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
        )
        wall_start = time.perf_counter()
        try:
            response = stub.Generate(request, timeout=args.rpc_timeout)
        except grpc.RpcError as exc:  # pragma: no cover - surface RPC errors to caller
            wall_end = time.perf_counter()
            return {
                "ok": False,
                "request_id": request_id,
                "error": exc.details() or str(exc),
                "latency_ms": (wall_end - wall_start) * 1000.0,
            }
        wall_end = time.perf_counter()
        return {
            "ok": True,
            "request_id": request_id,
            "latency_ms": (wall_end - wall_start) * 1000.0,
            "ttft_ms": response.ttft_ms or response.time_ms,
            "tpot_ms": response.tpot_ms,
            "tokens_per_second": response.tokens_per_second,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "time_ms": response.time_ms,
        }

    successes: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    with GPUMonitor(args.gpu_index, args.gpu_sample_interval) as monitor:
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_run_single, i, next(prompts_cycle)) for i in range(args.requests)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result["ok"]:
                    successes.append(result)
                else:
                    failures.append(result)
        wall_end = time.perf_counter()
        gpu_summary = monitor.summary()

    total_wall_ms = (wall_end - wall_start) * 1000.0
    throughput_rps = (len(successes) / ((wall_end - wall_start) or 1e-6))

    if successes:
        latencies = sorted(item["latency_ms"] for item in successes)
        ttfts = sorted(item["ttft_ms"] for item in successes)
        tpots = sorted(item["tpot_ms"] for item in successes)
        token_rates = sorted(item["tokens_per_second"] for item in successes)
    else:
        latencies = ttfts = tpots = token_rates = []

    summary = {
        "total_requests": args.requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "wall_time_ms": total_wall_ms,
        "throughput_rps": throughput_rps,
        "latency_ms": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50) if latencies else 0.0,
            "p95": _percentile(latencies, 95) if latencies else 0.0,
        },
        "ttft_ms": {
            "avg": statistics.mean(ttfts) if ttfts else 0.0,
            "p50": _percentile(ttfts, 50) if ttfts else 0.0,
            "p95": _percentile(ttfts, 95) if ttfts else 0.0,
        },
        "tpot_ms": statistics.mean(tpots) if tpots else 0.0,
        "tokens_per_second": statistics.mean(token_rates) if token_rates else 0.0,
        "total_output_tokens": int(sum(item.get("output_tokens", 0) for item in successes)),
        "gpu": gpu_summary,
        "failures": failures,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Mini load generator for the gRPC inference service")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--requests", type=int, default=16, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--prompt", help="Inline prompt to use for every request")
    parser.add_argument("--prompt-file", type=Path, help="Path to a file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k parameter")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--rpc-timeout", type=float, default=90.0, help="Per-request RPC timeout in seconds")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for NVML stats")
    parser.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.5,
        help="Interval between GPU samples in seconds",
    )
    parser.add_argument("--report", type=Path, help="Optional path to write a JSON summary")
    args = parser.parse_args()

    summary = run_load(args)

    print("=== Load Test Summary ===")
    print(json.dumps(summary, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
