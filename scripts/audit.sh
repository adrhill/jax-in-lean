#!/usr/bin/env bash
# Fails if the Lean sources contain an escape hatch, if the build fails,
# if a bug reproducer no longer shows its bug on the pinned JAX,
# or if the shapes or dtypes of the model disagree with those of the pinned JAX.
set -euo pipefail
cd "$(dirname "$0")/.."

forbidden='\b(sorry|admit|native_decide|axiom|unsafe|implemented_by)\b'
if grep -rnE "$forbidden" --include="*.lean" --exclude-dir=.lake .; then
  echo "audit: forbidden token found" >&2
  exit 1
fi
lake build
uv run python scripts/bugs.py
uv run python scripts/shapes.py
echo "audit: ok"
