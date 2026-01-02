import subprocess
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    proto_path = repo_root / "services" / "grpc" / "text_generation.proto"
    out_dir = repo_root / "services" / "grpc"

    cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        "--experimental_allow_proto3_optional",
        f"-I{proto_path.parent}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_path),
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))
    print(f"Generated gRPC stubs in {out_dir}")


if __name__ == "__main__":
    main()
