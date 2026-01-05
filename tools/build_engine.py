#!/usr/bin/env python3
"""Build TensorRT-LLM engine from HuggingFace model."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_tensorrt_llm():
    """Verify TensorRT-LLM is available."""
    try:
        import tensorrt_llm
        return True
    except ImportError:
        print("ERROR: TensorRT-LLM not found.", file=sys.stderr)
        print("Install it from: https://github.com/NVIDIA/TensorRT-LLM", file=sys.stderr)
        return False


def convert_checkpoint(model_id: str, checkpoint_dir: Path, dtype: str = "float16"):
    """Convert HuggingFace model to TensorRT-LLM checkpoint."""
    print(f"Converting {model_id} to TensorRT-LLM checkpoint...")
    
    cmd = [
        "python",
        "-m", "tensorrt_llm.commands.convert_checkpoint",
        "--model_dir", model_id,
        "--output_dir", str(checkpoint_dir),
        "--dtype", dtype,
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Checkpoint saved to {checkpoint_dir}")


def build_trt_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    max_batch_size: int = 8,
    max_input_len: int = 2048,
    max_output_len: int = 512,
    max_beam_width: int = 1,
):
    """Build TensorRT engine from checkpoint."""
    print(f"Building TensorRT engine...")
    
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(output_dir),
        "--gemm_plugin", "float16",
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_output_len", str(max_output_len),
        "--max_beam_width", str(max_beam_width),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Engine saved to {output_dir}")


def create_triton_config(
    model_name: str,
    output_dir: Path,
    max_batch_size: int = 8,
    max_queue_delay_us: int = 5000,
):
    """Generate Triton model config."""
    config_content = f"""name: "{model_name}"
backend: "tensorrtllm"
max_batch_size: {max_batch_size}

dynamic_batching {{
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: {max_queue_delay_us}
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

parameters {{
  key: "gpt_model_type"
  value: {{ string_value: "v1" }}
}}

parameters {{
  key: "gpt_model_path"
  value: {{ string_value: "/models/{model_name}/1" }}
}}
"""
    
    config_path = output_dir.parent / "config.pbtxt"
    config_path.write_text(config_content)
    print(f"Triton config written to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT-LLM engine for Triton Inference Server"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., microsoft/phi-3-mini-4k-instruct)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for Triton model (e.g., models/phi3/1)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model precision (default: float16)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size (default: 8)",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=2048,
        help="Maximum input sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=512,
        help="Maximum output sequence length (default: 512)",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip checkpoint conversion (assumes checkpoint exists)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip engine build (just generate config)",
    )
    args = parser.parse_args()
    
    if not check_tensorrt_llm() and not (args.skip_checkpoint and args.skip_build):
        sys.exit(1)
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir.parent / "checkpoint"
    
    # Step 1: Convert checkpoint
    if not args.skip_checkpoint:
        convert_checkpoint(args.model, checkpoint_dir, args.dtype)
    
    # Step 2: Build TRT engine
    if not args.skip_build:
        build_trt_engine(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
        )
    
    # Step 3: Generate Triton config
    model_name = output_dir.parent.name
    create_triton_config(
        model_name=model_name,
        output_dir=output_dir,
        max_batch_size=args.max_batch_size,
    )
    
    print("\n" + "="*60)
    print("SUCCESS: TensorRT-LLM engine built successfully!")
    print("="*60)
    print(f"\nModel repository: {output_dir.parent.parent}")
    print(f"Model name: {model_name}")
    print(f"\nTo start Triton server:")
    print(f"  docker run --gpus all -p 8000:8000 -p 8001:8001 \\")
    print(f"    -v {output_dir.parent.parent.absolute()}:/models \\")
    print(f"    nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print(f"    tritonserver --model-repository=/models")


if __name__ == "__main__":
    main()
