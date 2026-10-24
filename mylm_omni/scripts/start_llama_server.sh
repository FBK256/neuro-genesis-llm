#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  ./scripts/start_llama_server.sh MODEL.gguf MMPROJ.gguf [PORT] [CTX_SIZE]

Environment:
  LLAMA_SERVER_BIN   llama-server executable (default: llama-server)

The listener is intentionally bound to 127.0.0.1 and this script does not enable server tools.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

server_bin="$(printenv LLAMA_SERVER_BIN 2>/dev/null || true)"
if [[ -z "$server_bin" ]]; then server_bin="llama-server"; fi
model_path="$1"
mmproj_path="$2"
port=""
ctx_size=""
if [[ $# -ge 3 ]]; then port="$3"; fi
if [[ $# -ge 4 ]]; then ctx_size="$4"; fi
if [[ -z "$port" ]]; then port="8080"; fi
if [[ -z "$ctx_size" ]]; then ctx_size="8192"; fi

[[ -f "$model_path" ]] || { echo "Model file not found: $model_path" >&2; exit 2; }
[[ -f "$mmproj_path" ]] || { echo "mmproj file not found: $mmproj_path" >&2; exit 2; }
command -v "$server_bin" >/dev/null 2>&1 || {
  echo "llama-server not found; set LLAMA_SERVER_BIN to its executable path." >&2
  exit 2
}

[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || {
  echo "PORT must be an integer in [1, 65535]" >&2
  exit 2
}
[[ "$ctx_size" =~ ^[0-9]+$ ]] && (( ctx_size >= 1024 && ctx_size <= 131072 )) || {
  echo "CTX_SIZE must be an integer in [1024, 131072]" >&2
  exit 2
}

echo "Starting a loopback-only llama.cpp server."
echo "Model: $model_path"
echo "mmproj: $mmproj_path"
echo "Endpoint: http://127.0.0.1:$port/v1"
echo "No shell/browser/MCP server tools are enabled by this launcher."
echo "Use an mmproj built for this exact model family/revision."

# Deliberately bind to loopback; do not add --tools or bind to 0.0.0.0.
exec "$server_bin" \
  -m "$model_path" \
  --mmproj "$mmproj_path" \
  --alias local-vlm \
  --host 127.0.0.1 \
  --port "$port" \
  --ctx-size "$ctx_size" \
  --parallel 1
