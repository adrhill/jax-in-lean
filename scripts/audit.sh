#!/usr/bin/env bash
# Fails if the Lean sources contain an escape hatch, if the build fails,
# or if a bug reproducer no longer shows its bug on the pinned JAX.
set -euo pipefail
cd "$(dirname "$0")/.."

forbidden='\b(sorry|admit|native_decide|axiom|unsafe|implemented_by)\b'
if grep -rnE "$forbidden" --include="*.lean" --exclude-dir=.lake .; then
  echo "audit: forbidden token found" >&2
  exit 1
fi
lake build
uv run python scripts/bugs.py
echo "audit: ok"
