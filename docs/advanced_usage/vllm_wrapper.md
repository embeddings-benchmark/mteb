## vLLM Wrapper

!!! note 
    vLLM currently only supports a limited number of models, with many model implementations having subtle differences compared to the default implementations in mteb. We are working on it. For full list of supported models you can refer to [vllm documentation](https://docs.vllm.ai/en/stable/models/supported_models/#pooling-models).


## Installation

Reference: https://docs.vllm.ai/en/latest/getting_started/installation/

=== "uv"
    ```bash
    uv pip install mteb[vllm]
    ```

## Usage

### vllm EngineArgs

vLLM has a large number of parameters; here are some commonly used ones:

:::mteb.models.vllm_wrapper.VllmWrapperBase

For all vLLM parameters, please refer to https://docs.vllm.ai/en/latest/configuration/engine_args/.

### Embedding models

:::mteb.models.vllm_wrapper.VllmEncoderWrapper

```python
import mteb
from mteb.models.vllm_wrapper import VllmEncoderWrapper

def get_results(model: str, tasks: list[str]):
    """Evaluate a model on specified MTEB tasks using vLLM for inference."""
    encoder = VllmEncoderWrapper(model=model)
    tasks = mteb.get_tasks(tasks=tasks)

    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )
    return results


if __name__ == "__main__":
    MODEL_NAME = "intfloat/e5-small"
    MTEB_EMBED_TASKS = ["STS12"]

    results = get_results(model=MODEL_NAME, tasks=MTEB_EMBED_TASKS)
    print(results)
```

### Rerank models

To use a cross encoder for reranking. The following code shows a two-stage run with the second stage reading results saved from the first stage.

:::mteb.models.vllm_wrapper.VllmCrossEncoderWrapper

```python
import tempfile

import mteb
from mteb.models.vllm_wrapper import VllmCrossEncoderWrapper


def get_results(model: str, tasks: list[str], languages: list[str]):
    """Evaluate a model on specified MTEB tasks using vLLM for inference."""
    cross_encoder = VllmCrossEncoderWrapper(model=model)

    with tempfile.TemporaryDirectory() as prediction_folder:
        bm25s = mteb.get_model("bm25s")
        eval_splits = ["test"]

        mteb_tasks = mteb.get_tasks(
            tasks=tasks, languages=languages, eval_splits=eval_splits
        )

        mteb.evaluate(
            bm25s,
            mteb_tasks,
            prediction_folder=prediction_folder,
            show_progress_bar=False,
            # don't save results for test runs
            cache=None,
            overwrite_strategy="always",
        )

        second_stage_tasks = []
        for task in mteb_tasks:
            second_stage_tasks.append(
                task.convert_to_reranking(
                    prediction_folder,
                    top_k=10,
                )
            )

        results = mteb.evaluate(
            cross_encoder,
            second_stage_tasks,
            show_progress_bar=False,
            cache=None,
        )
    return results


if __name__ == "__main__":
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MTEB_RERANK_TASKS = ["NFCorpus"]
    MTEB_RERANK_LANGS = ["eng"]

    results = get_results(
        model=MODEL_NAME, tasks=MTEB_RERANK_TASKS, languages=MTEB_RERANK_LANGS
    )
    print(results)
```

## vLLM is fast with

### Half-Precision Inference

By default, vLLM uses Flash Attention, which only supports float16 and bfloat16 but not float32. vLLM does not optimize inference performance for float32.

<img src="https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/docs/images/visualizations/half_precision_Inference.png">

X-axis: Throughput (request/s)
Y-axis: Latency, Time needed for one step (ms) <- logarithmic scale
The curve lower right is better ↘

The throughput using float16 is approximately four times that of float32.

!!! note 
    |--------+------+----------+----------|
    | Format | Bits | Exponent | Fraction |
    |--------+------+----------+----------|
    | float32 | 32 | 8 | 23 |
    | float16 | 16 | 5 | 10 |
    | bfloat16 | 16 | 8 | 7 |
    |--------+------+----------+----------|

    If the model weights are stored in float32:
    - VLLM uses float16 for inference by default to inference a float32 model, it will keep numerical precision in most cases, for it have retains relatively more Fraction bits. However, due to the smaller Exponent part (only 5 bits), some models (e.g., the Gemma family) may risk producing NaN. VLLM maintains a list models that may cause NaN values and uses bfloat16 for inference by default.
    - Using bfloat16 for inference avoids NaN risks because its Exponent part matches float32 with 8 bits. However, with only 7 Fraction bits, numerical precision decreases noticeably.
    - Using float32 for inference incurs no precision loss but is about four times slower than float16/bfloat16.
    If model weights are stored in float16 or bfloat16, vLLM defaults to using the original dtype for inference.
    Quantization: With the advancement of open-source large models, fine-tuning of larger models for tasks like embedding and reranking is increasing. Exploring quantization methods to accelerate inference and reduce GPU memory usage may become necessary.

### Unpadding

By default, Sentence Transformers pads all inputs in a batch to the length of the longest one, which is undoubtedly very inefficient. VLLM avoids the use of padding entirely during inference.

<img src="https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/docs/images/visualizations/unpadding.png">

X-axis: Throughput (request/s)
Y-axis: Latency, Time needed for one step (ms) <- logarithmic scale
The curve lower right is better ↘

Sentence Transformers suffers a noticeable drop in speed when handling requests with varied input lengths, whereas vLLM does not.

### Others

For models using bidirectional attention, such as BERT, VLLM offers a range of performance optimizations:
- Optimized CUDA kernels, including FlashAttention and FlashInfer integration
- CUDA Graphs and torch.compile support to reduce overhead and accelerate execution
- Support for tensor, pipeline, data, and expert parallelism for distributed inference
- Multiple quantization schemes—GPTQ, AWQ, AutoRound, INT4, INT8, and FP8—for efficient deployment
- Continuous batching of incoming requests to maximize throughput

For causal attention models, such as the Qwen3 reranker, the following optimizations are also applicable:
- Efficient KV cache memory management via PagedAttention
- Chunked prefill for improved memory handling during long-context processing
- Prefix caching to accelerate repeated prompt processing

VLLM’s optimizations are primarily designed for and most effective with causal language models (generative models). For full list of features you can refer to [vllm documentation](https://docs.vllm.ai/en/latest/features/)
