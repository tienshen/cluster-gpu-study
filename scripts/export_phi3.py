import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Export Phi-3 Mini from Hugging Face")
    parser.add_argument(
        "--model",
        default="microsoft/phi-3-mini-4k-instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--output-dir",
        default="models/phi-3-mini",
        help="Directory to save the model",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache dir")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the repository",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
