#!/usr/bin/env python
"""Create a DNA character-level tokenizer and upload it to HuggingFace Hub."""

import argparse

from bolinas.tokenizer.char import create_char_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a DNA character-level tokenizer and upload it to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo id to upload to (e.g. bolinas-dna/tokenizer-char).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tok = create_char_tokenizer()
    tok.push_to_hub(args.repo)


if __name__ == "__main__":
    main()
