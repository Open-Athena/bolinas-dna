#!/usr/bin/env bash
# Removes a git worktree created by setup-worktree.sh.
# Usage: ./scripts/remove-worktree.sh <branch-name> [--delete-branch]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_NAME="$(basename "$MAIN_REPO")"

usage() {
    echo "Usage: $0 <branch-name> [--delete-branch]"
    echo ""
    echo "Removes the worktree at ../${REPO_NAME}-<branch-name>/"
    echo ""
    echo "Options:"
    echo "  --delete-branch    Also delete the git branch after removing the worktree"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

BRANCH_NAME="$1"
DELETE_BRANCH=false

if [[ $# -ge 2 && "$2" == "--delete-branch" ]]; then
    DELETE_BRANCH=true
fi

if [[ -z "$BRANCH_NAME" ]]; then
    echo "Error: Branch name cannot be empty"
    usage
fi

WORKTREE_DIR="$(dirname "$MAIN_REPO")/${REPO_NAME}-${BRANCH_NAME}"

# Check if worktree exists
if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "Error: Worktree directory does not exist: $WORKTREE_DIR"
    exit 1
fi

# Verify it's actually a worktree
cd "$MAIN_REPO"
if ! git worktree list | grep -q "$WORKTREE_DIR"; then
    echo "Error: Directory exists but is not a git worktree: $WORKTREE_DIR"
    exit 1
fi

echo "Removing worktree at $WORKTREE_DIR"

# Remove the worktree
git worktree remove "$WORKTREE_DIR"

echo "Worktree removed successfully"

if [[ "$DELETE_BRANCH" == true ]]; then
    echo "Deleting branch '$BRANCH_NAME'..."
    if git branch -d "$BRANCH_NAME" 2>/dev/null; then
        echo "Branch '$BRANCH_NAME' deleted"
    else
        echo "Warning: Could not delete branch '$BRANCH_NAME' (may have unmerged changes)"
        echo "Use 'git branch -D $BRANCH_NAME' to force delete"
    fi
else
    echo ""
    echo "Branch '$BRANCH_NAME' was kept. To delete it:"
    echo "  git branch -d $BRANCH_NAME"
fi
