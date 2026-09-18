---
title: "OpenAI-Compatible APIs"
icon: lucide/plug
---

## OpenAI-Compatible API Wrappers

MTEB provides wrappers for connecting to any OpenAI-compatible API server via HTTP for embedding, reranking, and ColBERT-style multi-vector retrieval tasks. These wrappers work with:

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

    - **Embedding**: `vllm serve <model-name> --runner pooling --port 8000`
    - **Reranking**: `vllm serve <reranker-model> --runner pooling --port 8001`
    - **Token-level (ColBERT-style)**: `vllm serve <model-name> --runner pooling --pooler-config.task token_embed --port 8002`

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

=== "Token-level / ColBERT-style (OpenAIAPITokenEmbedWrapper)"
    ```python
    import mteb
    from mteb.models import OpenAIAPITokenEmbedWrapper

    # Connect to a vLLM server serving a late-interaction (ColBERT-style)
    # model. Unlike the two wrappers above, this returns a per-token
    # multi-vector embedding for each input instead of one fixed-size
    # vector, and scores retrieval candidates with MaxSim rather than
    # cosine/dot similarity.
    model = OpenAIAPITokenEmbedWrapper(
        endpoint_url="http://localhost:8002",
        model_name="BAAI/bge-m3",
        modalities=["text"],
        # BAAI/bge-m3 has no chat template, so pure-text requests must use
        # the plain `input` field rather than `messages` (see "Multimodal
        # inputs" below).
        use_chat_template=False,
    )

    # Evaluate on an MTEB retrieval task
    results = mteb.evaluate(
        model,
        mteb.get_task("NanoSciFactRetrieval"),
    )
    print(results)
    ```

## Multimodal inputs

All three wrappers accept image, audio, and video content alongside text, by sending it to vLLM's Chat Embeddings/Pooling APIs (a `messages` field, following the [vLLM pooling examples](https://docs.vllm.ai/en/latest/examples/pooling/)) or, for reranking, as `{"content": [...]}` blocks on `/v1/rerank`. This requires a vLLM server started with a model that actually supports that modality — for example:

```bash
# Multimodal embeddings
vllm serve Qwen/Qwen3-VL-Embedding-2B --runner pooling --max-model-len 8192

# Multimodal reranking (image + video)
vllm serve Qwen/Qwen3-VL-Reranker-2B --runner pooling --max-model-len 4096 \
    --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
    --chat-template examples/pooling/score/template/qwen3_vl_reranker.jinja

# Multimodal ColBERT-style (image + text)
vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096
```

```python
import mteb
from mteb.models import OpenAIAPIEncodeWrapper

encoder = OpenAIAPIEncodeWrapper(
    endpoint_url="http://localhost:8000",
    model_name="Qwen/Qwen3-VL-Embedding-2B",
    modalities=["text", "image"],
)

results = mteb.evaluate(
    encoder,
    mteb.get_task("VisRAGRetArxivQA"),
)
print(results)
```

`OpenAIAPIRerankWrapper` supports image and video, but not audio: vLLM's rerank/score content-part schema has no audio variant. `OpenAIAPIEncodeWrapper` and `OpenAIAPITokenEmbedWrapper` support all four modalities (text, image, audio, video). Video is re-encoded from decoded frames via [`torchcodec`](https://pypi.org/project/torchcodec/) (`pip install mteb[video]`); resampling video/audio uses `mteb.models.modality_collators.VideoCollator`/`AudioCollator`, and can be tuned via `fps`, `max_frames`, `num_frames`, `target_sampling_rate`, and `max_samples` constructor arguments.

!!! warning "`use_chat_template`"
    Image/audio/video content is always sent via `messages`, which vLLM renders through the model's **chat template**. Non-chat text-encoder models — e.g. `BAAI/bge-small-en-v1.5`, `BAAI/bge-m3` — don't define one and will reject `messages` requests with a 400 error (`"...default chat template is no longer allowed..."`).

    - `OpenAIAPITokenEmbedWrapper` defaults to `use_chat_template=True` (every request, including pure text, goes through `messages`), since it's vLLM-only.
    - `OpenAIAPIEncodeWrapper` also defaults to `use_chat_template=True`, but is commonly pointed at the real OpenAI API or non-chat vLLM models — neither supports `messages` for embeddings — so set `use_chat_template=False` for those; pure-text requests then use the plain `input` field instead.
    - `OpenAIAPIRerankWrapper` has no such flag: text-only rerank/score requests never use `messages` in the first place (they use the `query`/`documents` string fields), so this only matters for the two wrappers above.

## Live testing scripts

`scripts/serve_vllm_models.sh` and `scripts/test_openai_wrappers_live.py` in the MTEB repository are ready-to-run companions covering all three wrappers, text-only and multimodal, against real small MTEB tasks:

```bash
# terminal 1: start a server for one scenario
scripts/serve_vllm_models.sh text-token-embed

# terminal 2: run the matching scenario
python scripts/test_openai_wrappers_live.py text-token-embed
```

Run `scripts/serve_vllm_models.sh --help` for the full list of scenarios (`text-embed`, `multimodal-embed`, `text-rerank`, `multimodal-rerank`, `text-token-embed`, `multimodal-token-embed`).

## API Reference

:::mteb.models.openai_wrappers.OpenAIAPIEncodeWrapper

:::mteb.models.openai_wrappers.OpenAIAPIRerankWrapper

:::mteb.models.openai_wrappers.OpenAIAPITokenEmbedWrapper
