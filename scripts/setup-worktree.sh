#!/usr/bin/env bash
# Creates a git worktree with symlinked results directories for parallel development.
# Usage: ./scripts/setup-worktree.sh <branch-name>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_NAME="$(basename "$MAIN_REPO")"

usage() {
    echo "Usage: $0 <branch-name>"
    echo ""
    echo "Creates a git worktree at ../${REPO_NAME}-<branch-name>/"
    echo "with results directories symlinked to the main repository."
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

BRANCH_NAME="$1"

if [[ -z "$BRANCH_NAME" ]]; then
    echo "Error: Branch name cannot be empty"
    usage
fi

# Validate branch name (basic check for invalid characters)
if [[ ! "$BRANCH_NAME" =~ ^[a-zA-Z0-9_/-]+$ ]]; then
    echo "Error: Invalid branch name '$BRANCH_NAME'"
    echo "Branch names should only contain alphanumeric characters, underscores, hyphens, and forward slashes"
    exit 1
fi

WORKTREE_DIR="$(dirname "$MAIN_REPO")/${REPO_NAME}-${BRANCH_NAME}"

if [[ -e "$WORKTREE_DIR" ]]; then
    echo "Error: Target directory already exists: $WORKTREE_DIR"
    exit 1
fi

echo "Creating worktree for branch '$BRANCH_NAME' at $WORKTREE_DIR"

# Create the worktree with a new branch
cd "$MAIN_REPO"
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR"

echo ""
echo "Symlinking results directories..."

# Find all results directories under snakemake/
RESULTS_DIRS=$(find "$MAIN_REPO/snakemake" -type d -name "results" 2>/dev/null || true)

LINKED_COUNT=0
for RESULTS_DIR in $RESULTS_DIRS; do
    # Get relative path from main repo
    REL_PATH="${RESULTS_DIR#$MAIN_REPO/}"
    TARGET_PARENT="$WORKTREE_DIR/$(dirname "$REL_PATH")"
    TARGET_PATH="$WORKTREE_DIR/$REL_PATH"

    # Create parent directory if needed
    mkdir -p "$TARGET_PARENT"

    # Remove the directory if it was created by git (would be empty)
    if [[ -d "$TARGET_PATH" && ! -L "$TARGET_PATH" ]]; then
        rmdir "$TARGET_PATH" 2>/dev/null || true
    fi

    # Create symlink
    if [[ ! -e "$TARGET_PATH" ]]; then
        ln -s "$RESULTS_DIR" "$TARGET_PATH"
        echo "  Linked: $REL_PATH -> $RESULTS_DIR"
        LINKED_COUNT=$((LINKED_COUNT + 1))
    fi
done

echo ""
echo "Worktree setup complete!"
echo "  Location: $WORKTREE_DIR"
echo "  Branch: $BRANCH_NAME"
echo "  Results directories linked: $LINKED_COUNT"
echo ""
echo "Next steps:"
echo "  cd $WORKTREE_DIR"
echo "  uv sync  # Create fresh virtual environment"
