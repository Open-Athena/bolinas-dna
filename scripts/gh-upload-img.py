#!/usr/bin/env python3
"""Upload files to a GitHub Gist and get permanent URLs.

Looks up gist ID from git config ``assets.gist``, falling back to ``pr.gist``
(for ghpr interop). Creates a new secret gist if none is configured, and saves
the ID to ``assets.gist`` for future uploads.

Adapted from https://github.com/ryan-williams/git-helpers
(github/gh-upload-img.py) by Ryan Williams.
"""

import argparse
import sys
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_call, check_output

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gist_upload

CONFIG_KEY = "assets.gist"
FALLBACK_KEY = "pr.gist"


def _git_config_get(key: str) -> str | None:
    """Read a git config value, returning None if unset."""
    try:
        return (
            check_output(["git", "config", key], stderr=DEVNULL).decode().strip()
            or None
        )
    except (CalledProcessError, FileNotFoundError):
        return None


def _git_config_set(key: str, value: str) -> bool:
    """Set a git config value. Returns True on success."""
    try:
        check_call(["git", "config", key, value], stderr=DEVNULL)
        return True
    except (CalledProcessError, FileNotFoundError):
        return False


def _get_or_create_gist(
    gist_id: str | None = None, description: str = "Asset uploads"
) -> str | None:
    """Return an existing gist ID from git config, or create a new one."""
    if gist_id:
        return gist_id

    for key in (CONFIG_KEY, FALLBACK_KEY):
        gist_id = _git_config_get(key)
        if gist_id:
            print(f"# Using gist from {key}: {gist_id}", file=sys.stderr)
            return gist_id

    print("# Creating new gist for assets...", file=sys.stderr)
    gist_id = gist_upload.create_gist(description)
    if not gist_id:
        print("Error: Could not create gist", file=sys.stderr)
        return None

    print(f"# Created gist: {gist_id}", file=sys.stderr)
    if _git_config_set(CONFIG_KEY, gist_id):
        print(f"# Saved gist ID to git config {CONFIG_KEY}", file=sys.stderr)
    else:
        print(
            "# Warning: couldn't save gist ID to git config (not in a git repo?)",
            file=sys.stderr,
        )
    return gist_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload files to a GitHub Gist and get permanent URLs"
    )
    parser.add_argument("files", nargs="+", help="Files to upload")
    parser.add_argument("-a", "--alt", help="Alt text for markdown/img format")
    parser.add_argument(
        "-b", "--branch", default="assets", help="Branch name in gist (default: assets)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["url", "markdown", "img", "auto"],
        default="auto",
        help="Output format (default: auto — markdown for images, url for others)",
    )
    parser.add_argument(
        "-g", "--gist", help="Gist ID to use (creates new if not specified)"
    )
    args = parser.parse_args()

    gist_id = _get_or_create_gist(args.gist)
    if not gist_id:
        sys.exit(1)

    files = [(path, Path(path).name) for path in args.files]

    results = gist_upload.upload_files_to_gist(
        files,
        gist_id,
        branch=args.branch,
        commit_msg="Add assets",
    )

    for orig_name, _safe_name, url in results:
        print(gist_upload.format_output(orig_name, url, args.format, args.alt))

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
