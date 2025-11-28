# Running the Evaluation

This section contains documentation related to the runtime of the evaluation. How to pass arguments to the encoder, saving outputs and similar.

## Simple Example

Evaluating models in `mteb` typically takes the simple form:

=== "Python"
    ```python
    import mteb

    model = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
    tasks = mteb.get_task(["MultiHateClassification"], languages = ["ita", "dan"])

    results = mteb.evaluate(model, tasks=tasks)
    ```
=== "CLI"
    ```bash
    mteb run -t MultiHateClassification -m sentence-transformers/static-similarity-mrl-multilingual-v1 -l ita dan
    ```

!!! info "Compatibility with SentenceTransformers"
    `mteb` is designed to be compatible with `sentence-transformers` so you can also directly pass a `SentenceTransformer` or `CrossEncoder` to `mteb.evaluate`.

## Specifying the cache

By default `mteb` with save the results in cache folder located at `~/.cache/mteb`, however if you want to save the results in a specific folder you
can specify it as follows:

```python
cache = mteb.ResultCache(cache_path="~/.cache/mteb")
results = mteb.evaluate(model, tasks=tasks, cache=cache)
```

If you don't wish to run model which results already exist on the leaderboard you can download these simply by running:

```python
cache.download_from_remote()
results = mteb.evaluate(model, tasks=tasks, cache=cache)
```

### Tracking Carbon Emissions

`mteb` allows for easy tracking of carbon emission eq. using `codecarbon`. You simply need to install `mteb[codecarbon]` and enable co2 tracking:

=== "Python"
    ```python
    results = mteb.evaluate(tasks=tasks, co2_tracker=True)
    ```
=== "CLI"
    ```bash
    mteb run -t NanoArguAnaRetrieval -m sentence-transformers/static-similarity-mrl-multilingual-v1 --co2-tracker
    ```

## Passing in `encode` arguments

To pass in arguments to the model's `encode` function, you can use the encode keyword arguments (`encode_kwargs`):

```python
mteb.evaluate(model, tasks, encode_kwargs={"batch_size": 32})
```

### Running SentenceTransformer model with prompts

Prompts can be passed to the SentenceTransformer model using the `prompts` parameter. The following code shows how to use prompts with SentenceTransformer:

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("intfloat/multilingual-e5-small", prompts={"query": "Query:", "document": "Passage:"})
results = mteb.evaluate(model, tasks=tasks)
```

In prompts the key can be:

1. Prompt types (`passage`, `query`) - they will be used in reranking and retrieval tasks
2. Task type - these prompts will be used in all tasks of the given type
   1. `BitextMining`
   2. `Classification`
   3. `MultilabelClassification`
   4. `Clustering`
   5. `PairClassification`
   6. `Reranking`
   7. `Retrieval`
   8. `STS`
   9. `Summarization`
   10. `InstructionRetrieval`
3. Pair of task type and prompt type like `Retrieval-query` - these prompts will be used in all Retrieval tasks
4. Task name - these prompts will be used in the specific task
5. Pair of task name and prompt type like `NFCorpus-query` - these prompts will be used in the specific task


## Saving predictions

To save the predictions from a task simply set the `prediction_folder`:

=== "Python"
    ```python
    # ...
    results = mteb.evaluate(
        model,
        task,
        prediction_folder="model_predictions",
    )
    ```

=== "CLI"
    ```bash
    mteb run -t NanoArguAnaRetrieval -m sentence-transformers/static-similarity-mrl-multilingual-v1 --prediction-folder predictions
    ```

The file will now be saved to `"{prediction_folder}/{task_name}_predictions.json"` and contain the rankings for each query along with the model name and revision of the model that produced the result.

## Speeding up Evaluations

Evaluation in MTEB consists of three main components: 1) downloading the dataset, 2) encoding of the samples, and 3) the evaluation. Typically, the most notable bottlenecks are either in the encoding step or in the download step. Where this bottleneck is depends on the tasks that you evaluate on and the models that you evaluate.

If you find that any of our design decisions prevents you from running the evaluation efficiently, do feel free to create an [issue](https://github.com/embeddings-benchmark/mteb/issues).

!!! info
    In version `2.1.4` an [issue](https://github.com/embeddings-benchmark/mteb/pull/3518) was found that caused the GPU and CPU to idle during evaluation of retrieval/reranking tasks. We suggest upgrading to `mteb>=2.2.0`.

### Speeding up the Model

MTEB is an evaluation framework, and therefore, we try to avoid doing inference optimizations. However, here are a few tricks to speed up inference.

First of all, it is possible to pass directly to the model loader:
```python
import mteb

meta = mteb.get_model_meta("intfloat/multilingual-e5-small")

kwargs = dict(
    device="cuda", # use a gpu
    model_kwargs={"torch_dtype": "float16"}, # use low-precision
)

model = meta.load_model(**kwargs) # passed to model loader, e.g. SentenceTransformer
```

This e.g., allows you to use all the [inference optimization tricks](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) from sentence-transformers. However, in general, you can always pass `device`.

If you can't find the required keyword argument, a solution might be to extract the model:
```python
model = meta.load_model(**kwargs)
# extract model
sentence_trf_model = model.model

# optimizations:
sentence_trf_model.half() # half precision
```

A last option is to make a [custom implementation](./defining_the_model.md#using-a-custom-model) of the model. This way you have full flexibility of how the model handles the input.


### Speeding Download

The simplest way to speed up downloads is by using Huggingface's [`xet`](https://huggingface.co/blog/xet-on-the-hub). You can use this simply using:

```bash
pip install mteb[xet]
```

For one of the larger datasets, `MrTidyRetrieval` (~15 GB), we have seen speed-ups from ~40 minutes to ~30 minutes while using `xet`.
