import argparse
import time
from concurrent import futures
from pathlib import Path
import sys

import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTO_DIR = Path(__file__).resolve().parent
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

import text_generation_pb2 as text_generation_pb2
import text_generation_pb2_grpc as text_generation_pb2_grpc


DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50


class TextGenerationService(text_generation_pb2_grpc.TextGenerationServicer):
    def __init__(self, model, tokenizer, device, default_do_sample):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.default_do_sample = default_do_sample

    def Generate(self, request, context):
        prompt = request.prompt
        if not prompt:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt is required")

        max_new_tokens = request.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        temperature = request.temperature or DEFAULT_TEMPERATURE
        top_p = request.top_p or DEFAULT_TOP_P
        top_k = request.top_k or DEFAULT_TOP_K
        do_sample = request.do_sample if request.HasField("do_sample") else self.default_do_sample

        if max_new_tokens <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "max_new_tokens must be > 0")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if request.stop:
            stop_index = text.find(request.stop)
            if stop_index != -1:
                text = text[:stop_index]

        input_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(generated.shape[-1] - input_tokens)

        return text_generation_pb2.GenerateResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_ms=elapsed_ms,
        )


def load_model(args):
    model_id = args.model_path or args.hf_model
    if model_id is None:
        raise ValueError("Either --model-path or --hf-model must be provided")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.model_path),
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.model_path),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bfloat16 else None,
    )
    model.eval()

    device = torch.device(args.device)
    if device.type == "cuda":
        model.to(device)
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="gRPC service for text generation")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=50051, help="Bind port")
    parser.add_argument("--model-path", help="Local model path (exported from Hugging Face)")
    parser.add_argument("--hf-model", help="Hugging Face model id (downloads if missing)")
    parser.add_argument("--cache-dir", help="Hugging Face cache dir")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--bfloat16", action="store_true", help="Load model with bfloat16")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the repository",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling by default",
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text_generation_pb2_grpc.add_TextGenerationServicer_to_server(
        TextGenerationService(model, tokenizer, device, args.do_sample),
        server,
    )
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"gRPC server running on {args.host}:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
