---
date: 2025-09-22
categories:
  - releases
authors:
  - samoed
  - kenneth
  - isaac
links:
  - Documentation: whats_new.md#new-in-v20
description: >
  We are very happy to announce the release of MTEB v2. This release has something for everyone, including a more consistent interface, better typing, new documentation, and better support for multimodal models and non-embedding based retrieval systems. This blog post will introduce you to some of the new features, but before diving into the new features, let's start with an introduction to MTEB and why we started working on v2.
---
# Introducing MTEB v2: Evaluation of embedding and retrieval systems for more than just text

We are very happy to announce the release of [MTEB v2](https://github.com/embeddings-benchmark/mteb). This release has something for everyone, including a more consistent interface, better typing, new documentation, and better support for multimodal models and non-embedding based retrieval systems. This blog post will introduce you to some of the new features, but before diving into the new features, let's start with an introduction to MTEB and why we started working on v2.

<!-- more -->


### What is MTEB?

The Massive Text Embedding Benchmark (MTEB) was initially launched by [Muennighoff et al (2022)](https://arxiv.org/abs/2210.07316) as a benchmark that broadens the scope of existing retrieval benchmarks to cover a broad range of embedding tasks including classification, clustering, reranking and retrieval. It has since then seen multiple contributions, such as the open-science collaboration MMTEB ([Enevoldsen et al., 2024](https://huggingface.co/papers/2502.13595)) adding a broad range of language support, the image extension, MIEB ([Xiao et al., 2025](https://huggingface.co/papers/2504.10471)), and a plethora of other monolingual and domain-specific extensions. We extend our sincere thanks to all those who have contributed!


### What are the reasons for the changes?

While all of these contributions to the package naturally had a huge positive impact on the scope of `mteb`, they also led to a bloating of the package, making it both hard to maintain and expand. In addition, the development in the field called for broader support for evaluation of retrieval systems across multiple modalities. Supporting these changes in the long-term without continuously breaking backward compatibility required us to do a large-scale refactor of MTEB, to put it in a state where we could introduce new changes without introducing breaking changes. Our design philosophy for long-term usability and reproducibility of embedding benchmarks is further detailed in [Chung et al., 2025](https://arxiv.org/abs/2506.21182v1).


### So what has changed?

This section gives an overview of the new features added in v2. Below we give an overview of changes following by detailed examples. If you are hungering for more information, do check out the [documentation](https://embeddings-benchmark.github.io/mteb/)

- [Easier evaluation using `mteb.evaluate`](#easier-evaluation)
- [Easier caching and results loading using the `ResultCache`](#better-local-and-online-caching)
- [Support for multimodal evaluation](#multimodal-input-format)
- [Better support for CrossEncoders](#better-support-for-crossencoders)
- [Unified Retrieval, Reranking and instruction variants](#unified-retrieval-reranking-and-instruction-variants)
- [Search Interface](#search-interface)
- [New documentation](#new-documentation)
- [Descriptive statistics for all tasks](#descriptive-statistics)
- [Better support for error analysis, allowing you to save predictions](#saving-predictions)
- [Support datasets v4](#support-datasets-v4)
- Standardization of file names and typing across the library (PEP8 compatible)
- Consistent logging and progress bars
- [Upgrading from v1](#upgrading-from-v1)
  - [Replacing `mteb.MTEB`](#replacing-mtebmteb)
  - [Replacing `mteb.load_results()`](#replacing-mtebload_results)
  - [Converting model to new format](#converting-model-to-new-format)
  - [Reuploading datasets](#reuploading-datasets)
  - [Converting Reranking datasets to new format](#converting-reranking-datasets-to-new-format)


### Easier evaluation

Evaluations are now a lot easier using [`mteb.evaluate`](https://embeddings-benchmark.github.io/mteb/api/evaluation/#mteb.evaluate),

```py
model = mteb.get_model("model_name")  # load reference implementation if it exists
tasks = mteb.get_tasks(tasks=["taskname1", "taskname2"])

results = mteb.evaluate(model, tasks)
```

### Better local and online caching
The new [`mteb.ResultCache`](https://embeddings-benchmark.github.io/mteb/api/results/#mteb.cache.ResultCache) makes managing the cache notably easier:
```py
from mteb.cache import ResultCache

model = ...
tasks = ...

cache = ResultCache(cache_path="~/.cache/mteb")  # default

# simple evaluate with cache
results = mteb.evaluate(model, tasks, cache=cache)  # only runs if results not in cache
```

It allow you to access the online cache so you don't have to rerun existing models.

```py
# no need to rerun already public results
cache.download_from_remote()  # download the latest results from the remote repository
results = mteb.evaluate(model, tasks, cache=cache)
```

### Multimodal Input format

Models in mteb who implements the [`Encoder`](https://embeddings-benchmark.github.io/mteb/api/model/#mteb.models.EncoderProtocol) protocol now supports multimodal input With the model protocol roughly looking like so:

```py
class EncoderProtocol(Protocol): # simplified
    """The interface for an encoder in MTEB."""

    def encode(self, inputs: DataLoader[BatchedInput], ...) -> Array: ...
```
Not only does this allow more efficient loading using the torch dataloader, but it also allows keys for multiple modalities:

```py
batch_input: BatchedInput = {
    "text": list[str],
    "images": list[PIL.Image],
    "audio": list[list[audio]],  # upcoming
    # + optional fields such as document title
}
```

Where `text` is a batch of texts and `list[images]` is a batch for that texts. This e.g. allows markdown documents with multiple figures like so:

> As you see in the following figure [figure 1](image_1) there is a correlation between A and B.

<details>
<summary>More examples of inputs</summary>

1. **[`TextInput`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.TextInput)**
   For pure text inputs.

   ```python
   {"text": ["This is a sample text.", "Another text."]}
   ```
2. **[`CorpusInput`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.CorpusInput)**
   For corpus-style inputs with titles and bodies.

   ```python
   {
       "text": ["Title 1 Body 1", "Title 2 Body 2"],
       "title": ["Title 1", "Title 2"],
       "body": ["Body 1", "Body 2"],
   }
   ```
3. **[`QueryInput`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.QueryInput)**
   For query–instruction pairs, typically used in retrieval or question answering tasks. Queries and instructions are combined with the model's instruction template.

   ```python
   {
       "text": [
           "Instruction: Your task is to find document for this query. Query: What is AI?",
           "Instruction: Your task is to find term for definition. Query: Define machine learning.",
       ],
       "query": ["What is AI?", "Define machine learning."],
       "instruction": [
           "Your task is find document for this query.",
           "Your task is to find term for definition.",
       ],
   }
   ```
4. **[`ImageInput`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.ImageInput)**
   For visual inputs consisting of images.

   ```python
   {"image": [PIL.Image1, PIL.Image2]}
   ```
5. **[`MultimodalInput`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.MultimodalInput)**
   For combined text–image (multimodal) inputs.

   ```python
   {"text": ["This is a sample text."], "image": [PIL.Image1]}
   ```

</details>

However, this also allows no text, multi-image inputs (e.g. for PDFs). Overall this greatly expands the possible tasks that can now be evaluated in MTEB.
To see how to convert a legacy model see the [converting model](#converting-model-to-new-format) section.

### Better support for CrossEncoders

Also, we've introduced a new [`CrossEncoderProtocol`](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types._encoder_io.MultimodalInput) for cross-encoders and now all cross-encoders have better support for evaluation:

```python

class CrossEncoderProtocol(Protocol):
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        ...
    ) -> Array:
```

### Unified Retrieval, Reranking and instruction variants

The retrieval tasks in MTEB now supports both retrieval and reranking using the same base task. The main difference now that Reranking tasks should have `top_ranked` subset to be evaluated on.
New structure of retrieval tasks:
```python
dataset[subset][split] = {
    "corpus": ...,  # dataset
    "queries": ...,  # dataset
    "relevant_docs": ...,  # dict[query id][corpus id] = score
    "top_ranked": ...,  # dict[query id] = list[corpus id]
}
```

On HuggingFace this dataset should these subsets:

1. `Corpus` - the corpus to retrieve from. Monolingual name: `corpus`, multilingual name: `{subset}-corpus`. Can contain columns:
   1. `id`, `text`, `title` for text corpus
   2. `id`, `image`, (`text` optionally) for image or multimodal corpus
2. `Queries` - the queries to retrieve with. Monolingual name: `queries`, multilingual name: `{subset}-queries`.
   1. `id`, `text` for text queries. Where text can be str for single query or `list[str]` or [`Conversation`][mteb.types._encoder_io.ConversationTurn] for multi-turn dialogs queries.
   2. `id`, `text`, `instructions` for instruction retrieval/reranking tasks
   3. `id`, `image`, (`text` optionally) for image or multimodal queries
3. `Qrels` - the relevance judgements. Monolingual name: `qrels`, multilingual name: `{subset}-qrels`.
      `query-id`, `corpus-id`, `score` (int or float) for relevance judgements.
4. `Top Ranked` - the top ranked documents to rerank. Only for reranking tasks. Monolingual name: `top_ranked`, multilingual name: `{subset}-top_ranked`.
      `query-id`, `corpus-ids` (`list[str]`) - the top ranked documents for each query.

### Search Interface

To make it easier to use MTEB for search, we have added a simple search interface using the new [`SearchProtocol`](https://embeddings-benchmark.github.io/mteb/api/model/#mteb.models.SearchProtocol):

```py
class SearchProtocol(Protocol):
    """Interface for searching models."""

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
    ) -> None: ...

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: dict[str, Any],
        top_ranked: TopRankedDocumentsType | None = None,
    ) -> RetrievalOutputType: ...
```

We're automatically wrapping `Encoder` and `CrossEncoder` models support `SearchProtocol`. However, if your model needs a custom index you can implement this protocol directly, like done for [pylate models to support `PlaidIndex`](https://github.com/embeddings-benchmark/mteb/blob/3effff65356b692a49e48c8f5c33680edd4e33b5/mteb/models/model_implementations/pylate_models.py#L31).

### New Documentation

We've added a lot of new documentation to make it easier to get started with MTEB.

- You can see api of our models in tasks in [API documentation](https://embeddings-benchmark.github.io/mteb/api/).
- We've added a [getting started guide](https://embeddings-benchmark.github.io/mteb/) to help you get started with MTEB.
- You can see implemented [tasks](https://embeddings-benchmark.github.io/mteb/overview/available_tasks/retrieval/) and [models](https://embeddings-benchmark.github.io/mteb/overview/available_models/text/) in MTEB.

### Better support for loading and comparing results

The new [`ResultCache`](https://embeddings-benchmark.github.io/mteb/api/results/#mteb.cache.ResultCache) also makes it easier to load, inspect and compare both local and online results:

```py
from mteb.cache import ResultCache

cache = ResultCache(cache_path="~/.cache/mteb")  # default
cache.download_from_remote()  # download the latest results from the remote repository

# load both local and online results
results = cache.load_results(
    models=["sentence-transformers/all-MiniLM-L6-v2", ...], tasks=["STS12"]
)
df = results.to_dataframe()
```

### Descriptive Statistics

Descriptive statistics isn't a new thing in MTEB, however, now it is there for every task, to extract it simply run:

```py
import mteb

task = mteb.get_task("MIRACLRetrievalHardNegatives")

task.metadata.descriptive_stats
```

And you will get a highly detailed set of descriptive statistics covering everything from number of samples query lengths, duplicates, etc. These not only make it easier for you to examine tasks, but it also makes it easier for us to make quality checks on future tasks.

Example for reranking task:
```json
{
    "test": {
        "num_samples": 160,
        "number_of_characters": 310133,
        "documents_text_statistics": {
            "total_text_length": 307938,
            "min_text_length": 0,
            "average_text_length": 2199.557142857143,
            "max_text_length": 2710,
            "unique_texts": 140
        },
        "documents_image_statistics": null,
        "queries_text_statistics": {
            "total_text_length": 2195,
            "min_text_length": 55,
            "average_text_length": 109.75,
            "max_text_length": 278,
            "unique_texts": 20
        },
        "queries_image_statistics": null,
        "relevant_docs_statistics": {
            "num_relevant_docs": 60,
            "min_relevant_docs_per_query": 7,
            "average_relevant_docs_per_query": 3.0,
            "max_relevant_docs_per_query": 7,
            "unique_relevant_docs": 140
        },
        "top_ranked_statistics": {
            "num_top_ranked": 140,
            "min_top_ranked_per_query": 7,
            "average_top_ranked_per_query": 7.0,
            "max_top_ranked_per_query": 7
        }
    }
}
```

Documentation for [the descriptive statistics types](https://embeddings-benchmark.github.io/mteb/api/types/#mteb.types.statistics).

### Saving Predictions

To support error analysis it is now possible to save the model prediction on a given task. You can do this simply as follows:
```python
import mteb

# using a small model and small dataset
encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
task = mteb.get_task("NanoArguAnaRetrieval")

prediction_folder = "path/to/model_predictions"

res = mteb.evaluate(
    encoder,
    task,
    prediction_folder=prediction_folder,
)
```

Result of prediction will be saved in `path/to/model_predictions/{task_name}_predictions.json` and will look like so for retrieval tasks:

```json
{
  "test": {
        "query1": {"document1": 0.77, "document2": 0.12, ...},
        "query2": {"document2": 0.87, "document1": 0.32, ...},
        ...
    }
}
```

### Support datasets v4

With the new functionality for [reuploading datasets](https://embeddings-benchmark.github.io/mteb/api/task/?h=push_dataset_to_hub#mteb.AbsTask.push_dataset_to_hub) to the standard datasets Parquet format, we’ve reuploaded all tasks with `trust_remote_code`, and MTEB now fully supports Datasets v4.

## Upgrading from v1

This section gives an introduction of how to upgrade from v1 to v2.

### Replacing `mteb.MTEB`

The previous approach to evaluate would require you to first create `MTEB` object and then call `.run` on that object.
The `MTEB` object was initially a sort of catch all object intended for both filtering tasks, selecting tasks, evaluating and few other cases.

This overload of functionality made it hard to change. We have already for a while made it easier to filter and select tasks using `get_tasks` and `mteb.evaluate` now superseded `MTEB` as the method for evaluation.

```py
# Approach before 2.0.0:
eval = mteb.MTEB(tasks=tasks) # now throw a deprecation warning
results = eval.run(
    model,
    overwrite=True,
    encode_kwargs={},
    ...
)

# Recommended:
mteb.evaluate(
    model,
    tasks,
    overwrite_strategy="only-missing", # only rerun missing splits
    encode_kwargs={},
    ...
)
```

### Replacing `mteb.load_results()`

Given the new `ResultCache` makes dealing with a results from _both_ local and online caches a lot easier, it can now replace `mteb.load_results` it

```py
tasks = mteb.get_tasks(tasks=["STS12"])
model_names = ["intfloat/multilingual-e5-large"]

# Approach before 2.0.0:
results = mteb.load_results(models=model_names, tasks=tasks, download_latest=True)

# Recommended:
cache = ResultCache("~/.cache/mteb")  # default
cache.download_from_remote()  # downloads remote results

results = cache.load_results(models=model_names, tasks=tasks)
```


### Converting model to new format

As mentioned in [the above section](#multimodal-input-format) MTEB v2, now supports multimodal input as the default.
Luckily for you all models implemented in MTEB already supports this new format! However, if you have a local model that you would like to evaluate.
Here is a quick conversion guide. If you previous implementation looks like so:

```py
# v1.X.X
class MyDummyEncoder:
    def __init__(self, **kwargs):
        self.model = ...

    def encode(self, sentences: list[str], **kwargs) -> Array:
        embeddings = self.model.encode(sentences)
        return embeddings
```

You can simply unpack it to its text input like so:

```py
# v2.0.0
class MyDummyEncoder:
    def __init__(self, **kwargs):
        self.model = ...

    def encode(self, input: DataLoader[BatchedInput], **kwargs) -> Array:
        # unpack to v1 format:
        sentences = [text for batch in inputs for text in batch["text"]]
        # do as you did beforehand:
        embeddings = self.model.encode(sentences)
        return embeddings
```

Of course, it will be more efficient if you work directly with the dataloader.

### Reuploading datasets

If your dataset is in old format, or you want to reupload it to the new Parquet format, you can do so using the new [`push_dataset_to_hub`](https://embeddings-benchmark.github.io/mteb/api/task/?h=push_dataset_to_hub#mteb.AbsTask.push_dataset_to_hub) method:

```py
import mteb

task = mteb.get_task("MyOldTask")
task.push_dataset_to_hub("my-username/my-new-task")
```

### Converting Reranking datasets to new format

If you have a reranking dataset, you can convert it to the retrieval format. To do this you need to add your task name to the `mteb.abstasks.text.reranking.OLD_FORMAT_RERANKING_TASKS`
and after this it would be converted to the new format automatically. To reupload them in new reranking format you refer to the [reuploading datasets](#reuploading-datasets) section.

```py
import mteb
from mteb.abstasks.text.reranking import OLD_FORMAT_RERANKING_TASKS

OLD_FORMAT_RERANKING_TASKS.append("MyOldRerankingTask")

task = mteb.get_task("MyOldRerankingTask")
model = ...
mteb.evaluate(model, task)
```
