#!/usr/bin/env bash
# Fails if the Lean sources contain an escape hatch, if the build fails,
# if the model depends on a non-standard axiom or disagrees with the registry of `Audit.lean`,
# if a bug reproducer no longer shows its bug on the pinned JAX,
# or if the shapes or dtypes of the model disagree with those of the pinned JAX.
set -euo pipefail
cd "$(dirname "$0")/.."

# `Audit.lean` finds what a proof depends on. This scan is for what leaves no trace there.
forbidden='\b(sorry|admit|native_decide|axiom|unsafe|implemented_by|skipKernelTC)\b'
if grep -rnE "$forbidden" --include="*.lean" --exclude-dir=.lake .; then
  echo "audit: forbidden token found" >&2
  exit 1
fi
lake build
lake env lean Audit.lean
uv run python scripts/bugs.py
uv run python scripts/shapes.py
echo "audit: ok"
