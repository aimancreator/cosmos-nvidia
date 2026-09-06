#!/bin/zsh
set -e
cd "${0:A:h}"
export UV_CACHE_DIR=.uv-cache
uv sync
exec uv run python flask_app.py

