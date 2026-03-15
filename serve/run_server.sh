#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash serve/run_server.sh /path/to/model.pt [--host 0.0.0.0] [--port 8000] [--workers 1] [--device 0] [--log-level info]

Example:
  bash serve/run_server.sh ../save/humanml_trans_dec_512_bert/model000200000.pt --port 9000 --device 0
USAGE
}

MODEL_PATH=""
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
LOG_LEVEL="info"
DEVICE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="${2:-}"
      shift 2
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "$MODEL_PATH" ]]; then
        MODEL_PATH="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Model path is required as the first argument." >&2
  usage
  exit 1
fi

export MDM_MODEL_PATH="$MODEL_PATH"
export MDM_DEVICE="$DEVICE"

exec uvicorn serve.generate_server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL"
