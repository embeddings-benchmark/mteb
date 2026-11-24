---
title: "Evaluation"
icon: lucide/drafting-compass
---

# Evaluation

:::mteb.evaluate

## Multiprocess execution configuration

`MultiProcessEncoderConfig` defines settings for one encoder worker per device.
Use `MultiProcessEncoder` to run dense text encoding with one persistent model
replica per device. Workers load models independently and return CPU NumPy
arrays in the original input order. Tasks, embedding caches, and retrieval
wrappers can use the existing encoder interface.

```python
import mteb

config = mteb.MultiProcessEncoderConfig(
    devices=["cuda:0", "cuda:1"],
    chunk_size=2048,
)
```

`chunk_size` controls the number of inputs in a dispatched job. Encoding
`batch_size` remains the inference batch size within each worker. Device
settings belong to execution configuration rather than model experiment kwargs
or embedding cache identities. Constructing the configuration does not load a
model or check device availability.

:::mteb.MultiProcessEncoderConfig

Cross-encoder prediction has an equivalent configuration. Its `chunk_size`
counts query-document pairs; prediction `batch_size` remains the number of
pairs in an inference batch within each worker.

```python
config = mteb.MultiProcessCrossEncoderConfig(
    devices=["cuda:0", "cuda:1"],
    chunk_size=2048,
)
```

:::mteb.MultiProcessCrossEncoderConfig

## Running workers

Run multiprocessing from a Python script with a main guard. The model loader
must be importable by spawned processes. Each device must fit a complete model
replica. Use a context manager to stop workers when evaluation finishes or
raises an error.

```python
import mteb


def main():
    meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
    with mteb.MultiProcessEncoder.from_model_meta(
        meta,
        devices=["cuda:0", "cuda:1"],
        chunk_size=2048,
    ) as model:
        results = mteb.evaluate(
            model,
            mteb.get_tasks(tasks=["STSBenchmark"]),
            encode_kwargs={"batch_size": 64},
        )


if __name__ == "__main__":
    main()
```

For cross-encoder evaluation, use `MultiProcessCrossEncoder.from_model_meta`
with cross-encoder metadata instead. Its `predict` method pairs the two
dataloaders by row and requires equal input counts. You can also construct
either wrapper directly with a model metadata object and its configuration:
`MultiProcessCrossEncoder(meta, config)`.

Workers are reused across encoding or prediction calls. At most one input
chunk per worker is in flight. Call methods sequentially. The encoder delegates
similarity and pairwise similarity to the loaded model on the first worker.
The `timeout` argument (default 300 seconds) limits worker startup and individual
jobs. A failed or timed-out worker closes the pool and propagates an error to
the caller; construct a new wrapper to retry. Outside a context manager, call
`close()` explicitly.

This implementation supports dense text encoding and text pair prediction.
Sparse, multi-vector, multimodal, and models using `device_map` are unsupported.
`num_proc` continues to control data loading separately from model workers.

:::mteb.MultiProcessEncoder

:::mteb.MultiProcessCrossEncoder
