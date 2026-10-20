#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
exec "$project_dir/mucap/.venv311/bin/python" -u -m multimodal_caption.mobile_pipeline --host "${MUCAP_HOST:-0.0.0.0}" --port "${MUCAP_PORT:-8765}"
