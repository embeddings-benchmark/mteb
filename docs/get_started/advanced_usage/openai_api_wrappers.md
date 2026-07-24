---
title: "OpenAI-Compatible APIs"
icon: lucide/plug
---

## OpenAI-Compatible API Wrappers

MTEB provides wrappers for connecting to any OpenAI-compatible API server via HTTP for embedding and reranking tasks. These wrappers work with:

- [vLLM](https://docs.vllm.ai/) servers
- [OpenAI](https://platform.openai.com/) APIs
- Any other server implementing the OpenAI-compatible `/v1/embeddings` or `/v1/rerank` endpoints

This is useful for:

- Benchmarking remote or production API servers
- Reusing running server instances across multiple benchmark runs
- Avoiding repeated model loading overhead
- Using hosted embedding and reranking APIs

!!! note "CLI support"
    The MTEB CLI does not currently support OpenAI-compatible API wrappers. Use the Python API directly as shown in the examples below.

## Usage

!!! note
    For vLLM, start a server with:

    - **Embedding**: `vllm serve <model-name> --host 0.0.0.0 --port 8000`
    - **Reranking**: `vllm serve <reranker-model> --host 0.0.0.0 --port 8001`

=== "Embedding models (OpenAIAPIEncodeWrapper)"
    ```python
    import mteb
    from mteb.models import OpenAIAPIEncodeWrapper

    # Connect to a vLLM server
    encoder = OpenAIAPIEncodeWrapper(
        endpoint_url="http://localhost:8000",
        model_name="BAAI/bge-small-en-v1.5",
    )

    # Or use OpenAI's API
    encoder = OpenAIAPIEncodeWrapper(
        endpoint_url="https://api.openai.com/v1",
        model_name="text-embedding-3-small",
        api_key="sk-...",
    )

    # Evaluate on MTEB tasks
    results = mteb.evaluate(
        encoder,
        mteb.get_task("STS12"),
    )
    print(results)
    ```

=== "Reranking models (OpenAIAPIRerankWrapper)"
    ```python
    import mteb
    from mteb.models import OpenAIAPIRerankWrapper

    # Connect to a vLLM reranking server
    reranker = OpenAIAPIRerankWrapper(
        endpoint_url="http://localhost:8001",
        model_name="BAAI/bge-reranker-v2-m3",
    )

    # Evaluate on MTEB reranking tasks
    results = mteb.evaluate(
        reranker,
        mteb.get_task("AskUbuntuDupQuestions"),
    )
    print(results)
    ```

## API Reference

:::mteb.models.openai_wrappers.OpenAIAPIEncodeWrapper

:::mteb.models.openai_wrappers.OpenAIAPIRerankWrapper
