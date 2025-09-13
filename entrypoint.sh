#!/usr/bin/env bash
set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

case "$1" in
  web)
    exec streamlit run app/streamlit_app.py --server.port "${STREAMLIT_SERVER_PORT:-8501}" --server.address "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
    ;;
  cli)
    shift
    exec python run.py "$@"
    ;;
  eval)
    shift
    exec python utils/evaluator.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
