import argparse
from pathlib import Path
import sys

import grpc

PROTO_DIR = Path(__file__).resolve().parent
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

import text_generation_pb2 as text_generation_pb2
import text_generation_pb2_grpc as text_generation_pb2_grpc


def main():
    parser = argparse.ArgumentParser(description="gRPC client for text generation")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--stop", default="", help="Stop sequence")
    args = parser.parse_args()

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = text_generation_pb2_grpc.TextGenerationStub(channel)
    request = text_generation_pb2.GenerateRequest(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        stop=args.stop,
    )
    response = stub.Generate(request)
    print(response.text)
    print(
        " ".join(
            [
                f"input_tokens={response.input_tokens}",
                f"output_tokens={response.output_tokens}",
                f"time_ms={response.time_ms:.2f}",
                f"ttft_ms={response.ttft_ms:.2f}",
                f"tpot_ms={response.tpot_ms:.3f}",
                f"tokens_per_second={response.tokens_per_second:.2f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
