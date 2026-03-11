"""Upload files to GitHub Gists via cloning, committing, and pushing.

Adapted from https://github.com/ryan-williams/git-helpers
(github/gist_upload.py, github/gh-upload-img.py) by Ryan Williams.
Modified to use HTTPS with gh auth credential helper, remove the local-clone
path, add type annotations, and simplify error handling.
"""

import mimetypes
import re
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from subprocess import (
    DEVNULL,
    PIPE,
    CalledProcessError,
    check_call,
    check_output,
    run,
)
from urllib.parse import quote

err = partial(print, file=sys.stderr)


def _run_quiet(cmd: list[str], **kwargs) -> None:
    """Run a command, suppressing stdout/stderr unless it fails."""
    result = run(cmd, stdout=PIPE, stderr=PIPE, **kwargs)
    if result.returncode != 0:
        for stream in (result.stderr, result.stdout):
            text = stream.decode().strip() if stream else ""
            if text:
                err(text)
        raise CalledProcessError(result.returncode, cmd)


def _get_github_username() -> str | None:
    """Get the current GitHub username via gh CLI."""
    try:
        return check_output(["gh", "api", "user", "--jq", ".login"]).decode().strip()
    except (CalledProcessError, FileNotFoundError):
        return None


def create_gist(
    description: str = "Image assets",
    content: str = "# Image Assets\nThis gist stores image assets.\n",
) -> str | None:
    """Create a new secret gist and return its ID."""
    try:
        output = (
            check_output(
                ["gh", "gist", "create", "--desc", description, "-"],
                input=content.encode(),
            )
            .decode()
            .strip()
        )
        match = re.search(r"gist\.github\.com/(?:[^/]+/)?([a-f0-9]+)", output)
        if match:
            return match.group(1)
    except CalledProcessError as e:
        err(f"Error creating gist: {e}")
    return None


def upload_files_to_gist(
    files: list[tuple[str, str]],
    gist_id: str,
    branch: str = "assets",
    commit_msg: str = "Add assets",
    verbose: bool = True,
) -> list[tuple[str, str, str]]:
    """Upload files to a gist branch by cloning, committing, and pushing.

    Clones the gist into a temporary directory, copies files in, commits, and
    pushes. Uses ``gh auth git-credential`` for HTTPS authentication.

    Args:
        files: (source_path, target_name) pairs.
        gist_id: Gist ID to upload to.
        branch: Branch name in the gist repo.
        commit_msg: Git commit message.
        verbose: Print progress to stderr.

    Returns:
        (original_name, target_name, url) triples for each uploaded file.
    """
    if not gist_id:
        err("Error: No gist ID provided")
        return []

    user = _get_github_username()
    if not user:
        err("Error: Could not determine GitHub username")
        return []

    resolved_files: list[tuple[Path, str]] = []
    for source_path, target_name in files:
        src = Path(source_path)
        if not src.exists():
            err(f"Error: File not found: {source_path}")
            continue
        resolved_files.append((src.resolve(), target_name))

    if not resolved_files:
        return []

    tmp = Path(tempfile.mkdtemp(prefix="gist_"))
    try:
        gist_url = f"https://gist.github.com/{gist_id}.git"
        _run_quiet(
            [
                "git",
                "clone",
                "-c",
                "credential.helper=!gh auth git-credential",
                gist_url,
                str(tmp),
            ]
        )

        git = partial(_run_git, cwd=tmp)
        git_output = partial(_run_git_output, cwd=tmp)

        git(["config", "credential.helper", "!gh auth git-credential"])

        remote = git_output(["remote"]).split("\n")[0]

        # Check out existing remote branch, or create a new orphan branch
        try:
            _run_quiet(["git", "fetch", remote, branch], cwd=tmp)
            _run_quiet(["git", "checkout", "-b", branch, f"{remote}/{branch}"], cwd=tmp)
            if verbose:
                err(f"Using existing branch '{branch}'")
        except CalledProcessError:
            _run_quiet(["git", "checkout", "-b", branch], cwd=tmp)
            if verbose:
                err(f"Created branch '{branch}'")

        for abs_path, target_name in resolved_files:
            shutil.copy2(abs_path, tmp / target_name)
            git(["add", target_name])
            if verbose:
                err(f"Added {target_name}")

        status = git_output(["status", "--porcelain"])
        if status:
            git(["commit", "-m", commit_msg])
            _run_quiet(["git", "push", remote, branch], cwd=tmp)

        commit_hash = git_output(["rev-parse", "HEAD"])

        results: list[tuple[str, str, str]] = []
        for abs_path, target_name in resolved_files:
            encoded = quote(target_name)
            url = f"https://gist.githubusercontent.com/{user}/{gist_id}/raw/{commit_hash}/{encoded}"
            results.append((abs_path.name, target_name, url))
            if verbose:
                err(f"Uploaded: {url}")

        return results

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def format_output(
    filename: str, url: str, format_type: str = "auto", alt_text: str | None = None
) -> str:
    """Format a URL for display as plain URL, markdown image, or HTML img tag."""
    if not alt_text:
        alt_text = filename

    output_format = format_type
    if format_type == "auto":
        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".ico",
            ".bmp",
        }
        ext = Path(filename).suffix.lower()
        if ext in image_extensions:
            output_format = "markdown"
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            output_format = (
                "markdown" if mime_type and mime_type.startswith("image/") else "url"
            )

    if output_format == "markdown":
        return f"![{alt_text}]({url})"
    if output_format == "img":
        return f'<img alt="{alt_text}" src="{url}" />'
    return url


def _run_git(args: list[str], *, cwd: Path) -> None:
    check_call(["git", *args], cwd=cwd, stdout=DEVNULL, stderr=DEVNULL)


def _run_git_output(args: list[str], *, cwd: Path) -> str:
    return check_output(["git", *args], cwd=cwd, stderr=DEVNULL).decode().strip()
