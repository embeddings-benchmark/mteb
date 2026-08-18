#!/usr/bin/env bash
# Start a vLLM server for one of the OpenAIAPI*Wrapper test scenarios.
#
# Companion to scripts/test_openai_wrappers_live.py. Run one scenario at a
# time (each starts a foreground server on its own port), then in another
# terminal run the matching scenario in the Python script, e.g.:
#
#   scripts/serve_vllm_models.sh text-embed
#   # in another terminal:
#   python scripts/test_openai_wrappers_live.py text-embed
#
# Requires vLLM to be installed (`pip install vllm`) and a GPU for the larger
# models. Ctrl-C stops the server.
#
# Usage: scripts/serve_vllm_models.sh <scenario> [extra vllm serve args...]

set -euo pipefail

SCENARIO="${1:-}"
shift || true

usage() {
    cat <<'EOF'
Usage: scripts/serve_vllm_models.sh <scenario> [extra vllm serve args...]

Scenarios:
  text-embed              BAAI/bge-small-en-v1.5           (port 8000, /v1/embeddings)
  multimodal-embed        Qwen/Qwen3-VL-Embedding-2B       (port 8000, /v1/embeddings)
  text-rerank             BAAI/bge-reranker-v2-m3          (port 8001, /v1/rerank)
  multimodal-rerank       Qwen/Qwen3-VL-Reranker-2B        (port 8001, /v1/rerank; image+video)
  text-token-embed        BAAI/bge-m3                      (port 8002, /pooling, ColBERT-style)
  multimodal-token-embed  TomoroAI/tomoro-colqwen3-embed-4b (port 8002, /pooling, ColBERT-style)

Any extra arguments are passed through to `vllm serve` as-is, e.g.:
  scripts/serve_vllm_models.sh text-embed --gpu-memory-utilization 0.5
EOF
}

if [[ -z "${SCENARIO}" ]]; then
    usage
    exit 1
fi

case "${SCENARIO}" in
    text-embed)
        exec vllm serve BAAI/bge-small-en-v1.5 \
            --port 8000 \
            --runner pooling \
            "$@"
        ;;

    multimodal-embed)
        # Chat Embeddings API (messages field) is used for image inputs.
        exec vllm serve Qwen/Qwen3-VL-Embedding-2B \
            --port 8000 \
            --runner pooling \
            --max-model-len 8192 \
            "$@"
        ;;

    text-rerank)
        exec vllm serve BAAI/bge-reranker-v2-m3 \
            --port 8001 \
            --runner pooling \
            "$@"
        ;;

    multimodal-rerank)
        # Requires vLLM's example chat template for the Qwen3-VL reranker.
        # VLLM_REPO should point at a checkout of https://github.com/vllm-project/vllm
        # so the --chat-template path below resolves.
        VLLM_REPO="${VLLM_REPO:-.}"
        CHAT_TEMPLATE="${VLLM_REPO}/examples/pooling/score/template/qwen3_vl_reranker.jinja"
        if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
            echo "error: chat template not found at ${CHAT_TEMPLATE}" >&2
            echo "Set VLLM_REPO to a checkout of github.com/vllm-project/vllm." >&2
            exit 1
        fi
        exec vllm serve Qwen/Qwen3-VL-Reranker-2B \
            --port 8001 \
            --runner pooling \
            --max-model-len 4096 \
            --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
            --chat-template "${CHAT_TEMPLATE}" \
            "$@"
        ;;

    text-token-embed)
        exec vllm serve BAAI/bge-m3 \
            --port 8002 \
            --runner pooling \
            --pooler-config.task token_embed \
            "$@"
        ;;

    multimodal-token-embed)
        exec vllm serve TomoroAI/tomoro-colqwen3-embed-4b \
            --port 8002 \
            --max-model-len 4096 \
            "$@"
        ;;

    -h|--help|help)
        usage
        exit 0
        ;;

    *)
        echo "error: unknown scenario '${SCENARIO}'" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
