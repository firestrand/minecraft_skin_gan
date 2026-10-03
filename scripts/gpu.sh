#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
scripts/uv.sh run --locked --with-requirements requirements-gpu.txt skin-gan "$@"
