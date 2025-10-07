# Running the Evaluation

This section contains documentation related to the runtime of the evaluation. How to pass arguments to the encoder, saving outputs and similar.

## Simple Example

Evalauting models in `mteb` typically takes the simple form:

```python
model = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
tasks = mteb.get_task(["MultiHateClassification"], languages = ["ita", "dan"])

results = mteb.evaluate(model, tasks=tasks)
```

!!! info "Compatibility with SentenceTransformers
    `mteb` is designed to be compatible with `sentence-transformers` so you can also directly pass a `SentenceTransformer` or `CrossEncoder` to `mteb.evaluate`.

## Specifying the cache

By default `mteb` with save the results in cache folder located at `~/.cache/mteb`, however if you want to saving the results in a specific folder you
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

```python
results = mteb.evaluate(tasks=tasks, co2_tracker=True)
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
Python:

```python
# ...

results = mteb.evaluate(
    model,
    task,
    prediction_folder="model_predictions",
)
```

The file will now be saved to `"{prediction_folder}/{task_name}_predictions.json"` and contain the rankings for each query along with the model name and revision of the model that produced the result.

For the CLI you can set the  `--prediction-folder` flag:
```bash
mteb run -t NanoArguAnaRetrieval -m sentence-transformers/static-similarity-mrl-multilingual-v1 --prediction-folder predictions 
```


## Running Cross Encoders on Reranking

To use a cross encoder for reranking, you can directly use a `CrossEncoder` from SentenceTransformers. The following code shows a two-stage run with the second stage reading results saved from the first stage.

```python
from sentence_transformers import CrossEncoder

import mteb

encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
task = mteb.get_task("NanoArguAnaRetrieval")

prediction_folder = "model_predictions"

# stage 1: retrieval
res = mteb.evaluate(
    encoder,
    task,
    prediction_folder=prediction_folder,
)

# convert task to retrieval
task = task.convert_to_reranking(prediction_folder, top_k=100)

# stage 2: reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
cross_enc_results = mteb.evaluate(cross_encoder, task)

task.metadata.main_score # NCDG@10
res[0].get_score()  # 0.286
cross_enc_results[0].get_score() # 0.338
```


## Using Late Interaction Models

This section outlines how to use late interaction models for retrieval using a ColBert model:

```python
import mteb
from mteb.models.search_wrappers import SearchEncoderWrapper

colbert = mteb.get_model("colbert-ir/colbertv2.0")
task = mteb.get_task("NanoArguAnaRetrieval")

mdl = SearchEncoderWrapper(colbert)
mdl.corpus_chunk_size = 500

results = mteb.evaluate(mdl, task)
```
This implementation employs the MaxSim operation to compute the similarity between sentences. While MaxSim provides high-quality results, it processes a larger number of embeddings, potentially leading to increased resource usage. To manage resource consumption, consider lowering the `corpus_chunk_size` parameter.

## Speeding up Evaluations

Evaluation in MTEB consists of three main components: 1) downloading the dataet, 2) encoding of the samples, and 3) the evaluation. Typically, the most notable bottleneck are either in the encoding step or on the download step. 
Where this bottleneck is depends on the tasks that you evaluate on and the models that you evaluate.

If you find that any of our design decisions prevents you from running the evaluation efficiently do feel free to create an [issue](https://github.com/embeddings-benchmark/mteb/issues).


### Speeding up the Model

MTEB is an evaluation framework and therefore we try to avoid doing inference optimizations. However, here is a few tricks to speed up inference.

First of all it is possible to pass directly to the model loader:
```python
import mteb

meta = mteb.get_model_meta("intfloat/multilingual-e5-small")

kwargs = dict(
    device="cuda", # use a gpu
    model_kwargs={"torch_dtype": "float16"}, # use low-precision
)

model = meta.load_model(**kwargs) # passed to model loader, e.g. SentenceTransformer
```

This e.g. allows you to use all the [inference optimization tricks](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) from sentence-transformers. However in general you can always pass `device`.

If you can't the required keyword argument a solution might be to extract the model:
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


## Caching Embeddings To Re-Use Them

There are times you may want to cache the embeddings so you can re-use them. This may be true if you have multiple query sets for the same corpus (e.g. Wikipedia) or are doing some optimization over the queries (e.g. prompting, other experiments). You can setup a cache by using a simple wrapper, which will save the cache per task in the `<path_to_cache_dir>/<task_name>` folder:

```python
# define your task(s) and model above as normal
task = mteb.get_task("LccSentimentClassification")
model = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")

# wrap the model with the cache wrapper
from mteb.models.cache_wrapper import CachedEmbeddingWrapper
model_with_cached_emb = CachedEmbeddingWrapper(model, cache_path='path_to_cache_dir')
# run as normal
results = mteb.evaluate(model_with_cached_emb, tasks=[task])
```

If you want to directly access the cached embeddings (e.g. for subsequent analyses) follow this example:

```python
import numpy as np
from mteb.models.cache_wrapper import TextVectorMap

# Access the memory-mapped file and convert to array
vector_map = TextVectorMap("path_to_cache_dir/LccSentimentClassification")
vector_map.load(name="LccSentimentClassification")
vectors = np.asarray(vector_map.vectors)

# Remove all "placeholders" in the embedding cache
zero_mask = (vectors == 0).all(axis=1)
vectors = vectors[~zero_mask]
```
