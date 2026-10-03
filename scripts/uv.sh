#!/usr/bin/env bash
set -euo pipefail
# Pin the package manager as well as the application dependencies.
exec uvx --from uv==0.12.22 uv "$@"
