## vLLM Wrapper

!!! note 
    vLLM currently only supports a limited number of models, with many model implementations having subtle differences compared to the default implementations in mteb. We are working on it. For full list of supported models you can refer to [vllm documentation](https://docs.vllm.ai/en/stable/models/supported_models/#pooling-models).

## vLLM is fast with

For models that use bidirectional attention, such as BERT.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- CUDA Graphs & torch.compile for reduced overhead and accelerated execution
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Quantizations: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for efficient deployment
- Continuous batching of incoming requests to maximize throughput
- When requests of varying lengths are batched together, there is no need to pad all inputs to the length of the longest request.

For models that use causal attention, such as the Qwen3 reranker. The following optimization can also be used.
- Efficient management of attention key and value memory with PagedAttention
- Chunked prefill
- Prefix caching

For full list of features you can refer to [vllm documentation](https://docs.vllm.ai/en/latest/features/)

!!! note 
    vllm uses flash attention by default, which does not support fp32. Therefore, it defaults to using fp16 for inference on fp32 models. Testing has shown a relatively small drop in accuracy. You can manually opt for fp32, but inference speed will be very slow.

## Installation

Reference: https://docs.vllm.ai/en/latest/getting_started/installation/

=== "uv"
    ```bash
    uv pip install mteb[vllm]
    ```

## vllm EngineArgs

vLLM has a large number of parameters; here are some commonly used ones:

:::mteb.models.vllm_wrapper.VllmWrapperBase

For all vLLM parameters, please refer to https://docs.vllm.ai/en/latest/configuration/engine_args/.

## Embedding models

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

## Rerank models

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
