---
title: "Multi-GPU evaluation"
icon: lucide/cpu
---

Models that run through Sentence Transformers (dense `SentenceTransformer`, `SparseEncoder` and `MultiVectorEncoder` models) can encode on several GPUs. Each worker process holds its own copy of the model and encodes a share of the inputs. There are two ways to ask for it, and it works the same way for models from mteb's registry and for Sentence Transformers models you load yourself.

For details on the pool itself, see the Sentence Transformers documentation on [multi-process / multi-GPU encoding](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#multi-process-multi-gpu-encoding).

## Encoding with a multi-process pool

Create the mteb model first, start a pool on its underlying Sentence Transformers model, and pass the pool through `encode_kwargs`. Start the pool once and reuse it for all tasks.

=== "Dense"

    ```python
    import mteb

    model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])

    pool = model.model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 32, "pool": pool})
    finally:
        model.model.stop_multi_process_pool(pool)
    ```

=== "Sparse"

    ```python
    import mteb

    model = mteb.get_model("Linkup-Platform/linkup-sparseup-embed-v1")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])

    pool = model.model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 32, "pool": pool})
    finally:
        model.model.stop_multi_process_pool(pool)
    ```

=== "Multi-vector"

    ```python
    import mteb

    model = mteb.get_model("perplexity-ai/pplx-embed-v2-late-9b")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])

    pool = model.model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 32, "pool": pool})
    finally:
        model.model.stop_multi_process_pool(pool)
    ```

`model.model` is the underlying Sentence Transformers model of the mteb wrapper.

## Passing a list of devices

If you don't want to manage a pool, pass the devices directly. Sentence Transformers then starts a temporary pool for each `encode` call and stops it afterwards:

```python
import mteb

model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
tasks = mteb.get_tasks(tasks=["NFCorpus"])

mteb.evaluate(
    model,
    tasks=tasks,
    encode_kwargs={"batch_size": 32, "device": ["cuda:0", "cuda:1"]},
)
```

This is the simplest option, but starting a pool reloads the model in every worker, and that happens on every `encode` call (for example once for the queries and once for the corpus of a retrieval task). For large models, or for tasks with many encode calls, prefer an explicit pool. Multi-vector models encode batch by batch, so a list of devices would start a pool for each batch: use an explicit pool for them.

## Using Sentence Transformers models directly

You can wrap a model you loaded yourself and evaluate it the same way. Wrap it with the matching mteb wrapper **before** starting the pool:

=== "Dense"

    ```python
    import mteb
    from mteb.models import SentenceTransformerEncoderWrapper
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"
    )
    model = SentenceTransformerEncoderWrapper(st_model)

    pool = st_model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(
            model,
            tasks=mteb.get_tasks(tasks=["NFCorpus"]),
            encode_kwargs={"batch_size": 32, "pool": pool},
        )
    finally:
        st_model.stop_multi_process_pool(pool)
    ```

=== "Sparse"

    ```python
    import mteb
    from mteb.models import SparseEncoderWrapper
    from sentence_transformers import SparseEncoder

    st_model = SparseEncoder("Linkup-Platform/linkup-sparseup-embed-v1", device="cuda:0")
    model = SparseEncoderWrapper(st_model)

    pool = st_model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(
            model,
            tasks=mteb.get_tasks(tasks=["NFCorpus"]),
            encode_kwargs={"batch_size": 32, "pool": pool},
        )
    finally:
        st_model.stop_multi_process_pool(pool)
    ```

=== "Multi-vector"

    ```python
    import mteb
    from mteb.models import MultiVectorWrapper
    from sentence_transformers import MultiVectorEncoder

    st_model = MultiVectorEncoder("perplexity-ai/pplx-embed-v2-late-9b", device="cuda:0")
    model = MultiVectorWrapper(st_model)

    pool = st_model.start_multi_process_pool(["cuda:0", "cuda:1"])
    try:
        mteb.evaluate(
            model,
            tasks=mteb.get_tasks(tasks=["NFCorpus"]),
            encode_kwargs={"batch_size": 32, "pool": pool},
        )
    finally:
        st_model.stop_multi_process_pool(pool)
    ```

## Where scoring runs

`start_multi_process_pool` (and a temporary pool created from a list of devices) moves the model in the main process to the CPU; only the workers use the GPUs. Encoding runs on the workers and the embeddings come back on the CPU.

Similarity between queries and documents is computed in the main process. mteb records the device the model was on when the wrapper was created and scores there, moving the document embeddings over in blocks so a large corpus chunk doesn't run out of memory.

!!! warning "Create the mteb model before starting the pool"
    If you start the pool first and wrap the model afterwards, the model is already on the CPU and similarity is scored on the CPU, which is very slow for large models and corpora. mteb warns when it sees a model on the CPU while a GPU (or MPS) is available. Create the mteb model first, as in the examples above.
