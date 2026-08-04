---
title: "vLLM Wrapper"
icon: lucide/zap
---


## vLLM

!!! note
    vLLM currently supports only a limited number of models, and many implementations have subtle differences compared to the default implementations in mteb. For the full list of supported models, refer to the [vllm documentation](https://docs.vllm.ai/en/stable/models/supported_models/#pooling-models).


## Installation

If you're using cuda you can run
=== "pip"
    ```bash
    pip install "mteb[vllm]"
    ```
=== "uv"
    ```bash
    uv pip install "mteb[vllm]"
    ```

For other architectures, please refer to the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

## Usage

To use vLLM with MTEB, you need to wrap the model with its corresponding wrapper class.

!!! warning "Python Multiprocessing Note"
    You **must** guard vLLM usage inside an `if __name__ == '__main__':` block to avoid Python multiprocessing issues. For example, instead of:

    ```python
    import vllm

    llm = vllm.LLM(...)
    ```

    do:

    ```python
    if __name__ == "__main__":
        import vllm

        llm = vllm.LLM(...)
    ```

    See the [vLLM troubleshooting guide](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing) for more details.

=== "Embedding models"
    ```python
    import mteb
    from mteb.models.vllm_wrapper import VllmEncoderWrapper


    def run_vllm_encoder():
        """Evaluate a model on specified MTEB tasks using vLLM for inference."""
        encoder = VllmEncoderWrapper(model="intfloat/e5-small")
        return mteb.evaluate(
            encoder,
            mteb.get_task("STS12"),
        )


    if __name__ == "__main__":
        results = run_vllm_encoder()
        print(results)
    ```
=== "Reranking models"
    ```python
    import mteb
    from mteb.models.vllm_wrapper import VllmCrossEncoderWrapper


    def run_vllm_crossencoder():
        """Evaluate a model on specified MTEB tasks using vLLM for inference."""
        cross_encoder = VllmCrossEncoderWrapper(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        return mteb.evaluate(
            cross_encoder,
            mteb.get_task("AskUbuntuDupQuestions"),
        )


    if __name__ == "__main__":
        results = run_vllm_crossencoder()
        print(results)
    ```

## Why is vLLM Fast?

### Half-Precision Inference

By default, vLLM uses Flash Attention, which only supports `float16` and `bfloat16`, not `float32`.

We provide a standalone benchmark script `scripts/bench_vllm/dtype.py` to quantify inference performance across different precisions.

<figure markdown="span">
    ![](../images/visualizations/half_precision_inference.png)
    <figcaption>Throughput with float16 is roughly 4× that of float32.<br>
    ST: Sentence Transformers backend; vLLM: vLLM backend.<br>
    X-axis: Throughput (requests/s); Y-axis: Latency (ms per step, log scale).<br>
    The lower‑right curve (↘) is better.</figcaption>
</figure>

!!! info "Floating‑Point Formats"

    | Format   | Bits | Exponent | Fraction |
    |----------|------|----------|----------|
    | float32  | 32   | 8        | 23       |
    | float16  | 16   | 5        | 10       |
    | bfloat16 | 16   | 8        | 7        |

    - When model weights are stored in `float32`, vLLM defaults to `float16` for inference. This generally preserves numerical precision well because `float16` keeps relatively more fraction bits, but due to its smaller exponent (5 bits), some models (e.g., the Gemma family) may produce NaNs. vLLM maintains a list of such models and uses `bfloat16` for them by default.
    - Using `bfloat16` avoids NaN risks because its exponent matches `float32` (8 bits), but with only 7 fraction bits, numerical precision degrades noticeably.
    - Using `float32` incurs no precision loss but is roughly 4× slower than half‑precision (`float16`/`bfloat16`).

    If model weights are natively in `float16` or `bfloat16`, vLLM defaults to the original dtype for inference.

    **Quantization**: With the rise of open‑source large models, fine‑tuned models for embedding and reranking are becoming larger. Exploring quantization methods (GPTQ, AWQ, etc.) to accelerate inference and reduce GPU memory usage may become necessary.

### Unpadding

By default, Sentence Transformers (ST) pad all inputs in a batch to the length of the longest one, which is highly inefficient. vLLM avoids padding entirely during inference.

We provide a standalone benchmark script `scripts/bench_vllm/unpadding.py` to quantify inference performance using unpadding.

<figure markdown="span">
    ![](../images/visualizations/unpadding.png)
    <figcaption>X-axis: Throughput (requests/s);<br>
    ST: Sentence Transformers; vLLM: vLLM.<br>
    Y-axis: Latency (ms per step, log scale).<br>
    The lower‑right curve (↘) is better.</figcaption>
</figure>

ST suffers a noticeable drop in speed when handling requests with varied input lengths, whereas vLLM does not.

### Overlap preprocessing and computation

(Available since vLLM 0.26.0)

For these small models, preprocessing bottlenecks are often encountered.

- Use multithreading to accelerate preprocessing. You can specify the number of threads using renderer_num_workers. The total time scales down almost linearly as the number of renderer workers increases, if you encounter preprocessing bottlenecks.
- Tiling to overlap preprocessing and computation for pooling models offline inference. When preprocessing takes less time than computation, the preprocessing overhead can be almost entirely overlapped.

We provide a standalone benchmark script `scripts/bench_vllm/renderer_num_workers.py` to quantify inference performance using renderer_num_workers.

<figure markdown="span">
    ![](../images/visualizations/renderer_workers.png)
    <figcaption>X‑axis: Prompt length (words, log₂ scale).<br>
    Y‑axis: Time for 100 embeddings (seconds, log₁₀ scale).<br>
    Each curve corresponds to a different number of renderer workers (1, 2, 4, 8).<br>
    Lower curves is better.</figcaption>
</figure>

### Other Optimizations

For models using bidirectional attention (e.g., BERT), vLLM offers a range of performance optimisations:

- Optimised CUDA kernels (integrating FlashAttention and FlashInfer)
- CUDA Graphs and `torch.compile` support to reduce overhead and accelerate execution
- Support for tensor, pipeline, data, and expert parallelism for distributed inference
- Multiple quantization schemes (GPTQ, AWQ, AutoRound, INT4, INT8, FP8) for efficient deployment
- Continuous batching of incoming requests to maximise throughput

For causal attention models (e.g., Qwen3 reranker), the following additional optimisations apply:

- Efficient KV cache memory management via PagedAttention
- Chunked prefill for improved memory handling during long‑context processing
- Prefix caching to accelerate repeated prompt processing

vLLM’s optimisations are primarily designed for and most effective with causal language models (generative models). For the full list of features, refer to the [vLLM features documentation](https://docs.vllm.ai/en/latest/features/).

## vLLM Pooling Models

### What are pooling models?

vLLM models can be categorized into two types:

- **[Generative Models](https://docs.vllm.ai/en/latest/models/supported_models/)** - Models that produce text completions or chat responses (e.g., LLaMA, Qwen, DeepSeek). Use `LLM.generate()` and `LLM.chat()` for these models.

- **[Pooling Models](https://docs.vllm.ai/en/latest/models/pooling_models/)** - These models do not generate content. They are primarily used for classification and retrieval tasks, such as bge-m3 and Qwen3 Reranker.

### Sequence-wise Task and Token-wise Task

The key distinction between sequence-wise task and token-wise task lies in their output granularity: sequence-wise task
produces a single result for an entire input sequence, whereas token-wise task yields a result for each individual token
within the sequence.

### Pooling Usages

| Pooling Usages                                                                                      | Description                                                                                                                                                                      |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Classification Usages](https://docs.vllm.ai/en/latest/models/pooling_models/classify/)             | Predicting which predefined category, class, or label best corresponds to a given input.                                                                                         |
| [Embedding Usages](https://docs.vllm.ai/en/latest/models/pooling_models/embed/)                     | Converts unstructured data (text, images, audio, etc.) into structured numerical vectors (embeddings).                                                                           |
| [Token Classification Usages](https://docs.vllm.ai/en/latest/models/pooling_models/token_classify/) | Token-wise classification                                                                                                                                                        |
| [Token Embedding Usages](https://docs.vllm.ai/en/latest/models/pooling_models/token_embed/)         | Token-wise embedding                                                                                                                                                             |
| [Reward Usages](https://docs.vllm.ai/en/latest/models/pooling_models/reward/)                       | Evaluates the quality of outputs generated by a language model, acting as a proxy for human preferences.                                                                         |
| [Scoring Usages](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/)                     | Computes similarity scores between two inputs. It supports three model types (aka `score_type`): `cross-encoder`, `late-interaction`, and `bi-encoder`.                          |
| Plugins Usages                                                                                      | Allow users to customize input and output processors. For more information, please refer to [IO Processor Plugins](https://docs.vllm.ai/en/latest/design/io_processor_plugins/). |

## API Reference

:::mteb.models.vllm_wrapper.VllmWrapperBase

!!! info "vLLM Engine Arguments"
    For all vLLM engine parameters, please refer to: https://docs.vllm.ai/en/latest/configuration/engine_args/.

:::mteb.models.vllm_wrapper.VllmEncoderWrapper

:::mteb.models.vllm_wrapper.VllmCrossEncoderWrapper
