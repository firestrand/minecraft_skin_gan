#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/uv.sh sync --locked --group dev
scripts/uv.sh run --locked ruff format --check .
scripts/uv.sh run --locked ruff check .
scripts/uv.sh run --locked ty check
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  scripts/uv.sh run --locked pytest
scripts/uv.sh run --locked python scripts/check_coverage.py
scripts/uv.sh build --no-sources --no-build-isolation
