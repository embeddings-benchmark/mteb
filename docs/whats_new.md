# What's New


## New since v2.3


### Overview of patch updates

- Make `PIL` optional as it is only used for type-checks ([#3713](https://github.com/embeddings-benchmark/mteb/pull/3713)) 
- Only use a shallow clone when download results to reduce download time and memory usage ([#3682](https://github.com/embeddings-benchmark/mteb/pull/3682))
- `get_model` now correctly assumed `SentenceTransformer` ([#3673](https://github.com/embeddings-benchmark/mteb/pull/3673))
- Fixed incorrect metadata on memory usage ([#3624](https://github.com/embeddings-benchmark/mteb/pull/3624))
- Resolved issue with laoding Vidore ([#3618](https://github.com/embeddings-benchmark/mteb/pull/3618))
- Updated to logging, such that "warning" is noted stated in the message if it is not a warning ([#3619](https://github.com/embeddings-benchmark/mteb/pull/3619))
- Added new metric (hamming score) to multilabel classification
  ([#3700](https://github.com/embeddings-benchmark/mteb/pull/3700))

- **Leaderboard**
    - Bumped gradio to v6 ([#3629](https://github.com/embeddings-benchmark/mteb/pull/3629))
    - Fixed display for task information and improve UI for benchmark filtering ([#3629](https://github.com/embeddings-benchmark/mteb/pull/3629))
    - Remove "Unknown" for int on leaderboard causing them to be unsortable ([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653))
    - Add optional "per language table" ([#3617](https://github.com/embeddings-benchmark/mteb/pull/3617))
    - Minor improvements to the leaderboard

- **Docs**
    - Update "speeding up"-section to include benefit gained from bumping the version of MTEB ([#3634](https://github.com/embeddings-benchmark/mteb/pull/3634))


- **Model fixes**
    - Fixed loading of GoogleTextEmbeddingModel which were given multiple `model_name`([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653))
    - Fixed loading of Linq-Embed-Mistral which does not take `instruction_template` as a kwargs ([#3653](https://github.com/embeddings-benchmark/mteb/pull/3653))
    - Fix bug in `openai/text-embedding-ada-002` caused by passing `embed_dim`
    ([#3689](https://github.com/embeddings-benchmark/mteb/pull/3689))

- **New models**
    - Added `FacebookAI/xlm-roberta-large` ([#3701](https://github.com/embeddings-benchmark/mteb/pull/3701))
    - Added `FacebookAI/xlm-roberta-base` ([#3701](https://github.com/embeddings-benchmark/mteb/pull/3701))
    - Added `KBLab/sentence-bert-swedish-cased` ([#3701](https://github.com/embeddings-benchmark/mteb/pull/3701))
    - Added `KFST/XLMRoberta-en-da-sv-nb` ([#3701](https://github.com/embeddings-benchmark/mteb/pull/3701))


## New in v2.3

### Support for custom search backends

MTEB v2.3 adds support for custom search encoder [IndexEncoderSearchProtocol](https://embeddings-benchmark.github.io/mteb/api/model/?h=indexencodersearchprotocol#mteb.models.IndexEncoderSearchProtocol) and adds the [FaissSearchIndex](https://embeddings-benchmark.github.io/mteb/advanced_usage/retrieval_backend/?h=faisssearchindex).

```python
import mteb
from mteb.models import SearchEncoderWrapper
from mteb.models.search_encoder_index import FaissSearchIndex

model = mteb.get_model(...)
index_backend = FaissSearchIndex(model)
model = SearchEncoderWrapper(
    model,
    index_backend=index_backend
)
...
```

This leads to a slight increase in performance, for example running `minishlab/potion-base-2M` on `SWEbenchVerifiedRR` took 694 seconds instead of 769. It, however, does not change the default behaviour.

### Overview of patch updates

- Fixed incorrect metadata on model memory ([#3624](https://github.com/embeddings-benchmark/mteb/pull/3624))
- Resolved issue with laoding Vidore ([#3618](https://github.com/embeddings-benchmark/mteb/pull/3618))
- Updated to logging, such that "warning" is noted stated in the message if it is not a warning ([#3619](https://github.com/embeddings-benchmark/mteb/pull/3619))

- **Docs**
    - Update "speeding up"-section to include benefit gained from bumping the version of MTEB ([#3634](https://github.com/embeddings-benchmark/mteb/pull/3634))


## New in v2.2

### Support for Assymetric embeddings in STS and `PairClassification`
MTEB v2.2 adds support for `prompt_type` for `STS` and `PairClassification` thus allowing for asymmetric embeddings.

E.g. for `TERRa`, this allow us to add `TERRa.v2`, 
```python
class TERRaV2(AbsTaskPairClassification):
    input1_prompt_type = PromptType.document
    input2_prompt_type = PromptType.query

    metadata = TaskMetadata(
        name="TERRa.V2", ...
    )
```

This is not backward compatible in scores for models with `query`/`document` separation, which is why we introduce the v2, but it better reflect the actual performance of these models.

Example for `intfloat/multilingual-e5-small`:

| Task | main | PR |
| - | - | - |
| TERRa.v2 | 0.575105 | 0.589083 |

### New Benchmark Vidore v3
Added Vidore V3 to the leaderboard ([#3542](https://github.com/embeddings-benchmark/mteb/pull/3542)), thanks [QuentinJGMace](https://github.com/QuentinJGMace) et al for working on this!

### Overview of patch updates

- Add support for python 3.14 ([#3450](https://github.com/embeddings-benchmark/mteb/pull/3450))
- Multiple fixes to leaderboard links, e.g.
  [#3591](https://github.com/embeddings-benchmark/mteb/pull/3591) and [#3560](https://github.com/embeddings-benchmark/mteb/pull/3560). Thanks [antoineedy](https://github.com/antoineedy)
- Resolve bugs where cache was ignored ([#3558](https://github.com/embeddings-benchmark/mteb/pull/3558)), thanks [dongwook92](https://github.com/dongwook92) for the PR
- Resolve hash randomization in retrieval task ID generation
  ([#3553](https://github.com/embeddings-benchmark/mteb/pull/3553))
- Improve messages for running missing splits
  ([#3596](https://github.com/embeddings-benchmark/mteb/pull/3596))
- Remove `set_float32_matmul_precision` as this lead to errors in some enviroments
  ([#3509](https://github.com/embeddings-benchmark/mteb/pull/3509))
- Add prompts to hardnegative tasks and bumping their version to maintain backward compatibility
  ([#3469](https://github.com/embeddings-benchmark/mteb/pull/3469))

- **Docs**
    - Minor fixes, e.g. updates of brokens links in the readme

- **Model fixes**
    - Correcting bug in the Cohere model implementation ([#3610](https://github.com/embeddings-benchmark/mteb/pull/3610))
    - Set default `input_type` for 'VoyageMultiModalModelWrapper' ([#3567](https://github.com/embeddings-benchmark/mteb/pull/3567))

- **New models**
    - Add jasper token compression model ([#3557](https://github.com/embeddings-benchmark/mteb/pull/3557))


## New in v2.1

### New benchmark for Dutch

MTEB v2.1 introduces a new benchmark for dutch `MTEB(nld, v1)` ([#3464](https://github.com/embeddings-benchmark/mteb/pull/3464)). Thanks to [nikolay-banar](https://github.com/nikolay-banar) for the PR.

### Overview of patch updates

- Speed up retrieval computation by more 6x ([#3454](https://github.com/embeddings-benchmark/mteb/pull/3454))
- Docs
    - Added model citations
    - spelling fixes 

## New in v2.0
This section goes through new features added in v2. Below we give an overview of changes following by detailed examples.

### Overview of changes

- New in v2.0
    - [Easier evaluation](#easier-evaluation)
    - [Better local and online caching](#better-local-and-online-caching)
    - [Multimodal Input format](#multimodal-input-format)
    - [Better support for CrossEncoders](#better-support-for-crossencoders)
    - [Unified Retrieval, Reranking and instruction variants](#unified-retrieval-reranking-and-instruction-variants)
    - [Search Interface](#search-interface)
    - [New Documentation](#new-documentation)
    - [Better support for loading and comparing results](#better-support-for-loading-and-comparing-results)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Saving Predictions](#saving-predictions)
    - [Support datasets v4](#support-datasets-v4)
  - [Upgrading from v1](#upgrading-from-v1)
    - [Replacing `mteb.MTEB`](#replacing-mtebmteb)
    - [Replacing `mteb.load_results()`](#replacing-mtebload_results)
    - [Converting model to new format](#converting-model-to-new-format)
    - [Reuploading datasets](#reuploading-datasets)
    - [Converting Reranking datasets to new format](#converting-reranking-datasets-to-new-format)

What are the reasons for the changes? Generally the many inconsistencies in the library made it hard to maintain without introducing breaking changes and we do think that there are multiple important areas to expand in, e.g. [adding new benchmark for image embeddings][@mieb_2025], support new model types in general making the library more accessible.
We have already been able to add many new feature in v2.0, but hope that this new version allow us to keep doing so without breaking backward compatibility. See [upgrading from v1](#upgrading-from-v1) for specific deprecations and how to fix them.


#### Easier evaluation

Evaluations are now a lot easier using [`mteb.evaluate`][mteb.evaluate.evaluate],

```py
results = mteb.evaluate(model, tasks)
```

#### Better local and online caching
The new [`mteb.ResultCache`][mteb.cache.ResultCache] makes managing the cache notably easier:
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
cache.download_from_remote() # download the latest results from the remote repository
results = mteb.evaluate(model, tasks, cache=cache)
```

#### Multimodal Input format

Models in mteb who implements the [`Encoder`][mteb.models.EncoderProtocol] protocol now supports multimodal input With the model protocol roughly looking like so:

```py
class EncoderProtocol(Protocol):  # simplified
    """The interface for an encoder in MTEB."""

    def encode(self, inputs: DataLoader[BatchedInput], ...) -> Array: ...
```
Not only does this allow more efficient loading using the torch dataloader, but it also allows keys for multiple modalities:

```py
batch_input: BatchedInput = {
    "text": list[str],
    "images": list[PIL.Image],
    "audio": list[list[audio]], # upcoming
    # + optional fields such as document title
}
```

Where `text` is a batch of texts and `list[images]` is a batch for that texts. This e.g. allows markdown documents with multiple figures like so:

> As you see in the following figure [figure 1](image_1) there is a correlation between A and B.

!!! Note
    More examples of new multimodal inputs you can find in [BatchedInput][mteb.types._encoder_io.BatchedInput] documentation.

However, this also allows no text, multi-image inputs (e.g. for PDFs). Overall this greatly expands the possible tasks that can now be evaluated in MTEB.
To see how to convert a legacy model see the [converting model](#converting-model-to-new-format) section.

#### Better support for CrossEncoders

Also, we've introduced a new [`CrossEncoderProtocol`][mteb.models.CrossEncoderProtocol] for cross-encoders and now all cross-encoders have better support for evaluation:

```python

class CrossEncoderProtocol(Protocol):
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        ...
    ) -> Array:
```

#### Unified Retrieval, Reranking and instruction variants

The retrieval tasks in MTEB now supports both retrieval and reranking using the same base task. The main difference now that Reranking tasks should have `top_ranked` subset to be evaluated on.
New structure of retrieval tasks: `dataset[subset][split]` = [RetrievalSplitData][mteb.abstasks.retrieval_dataset_loaders.RetrievalSplitData]. On HF this dataset should these subsets:

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

#### Search Interface

To make it easier to use MTEB for search, we have added a simple search interface using the new [`SearchProtocol`][mteb.models.SearchProtocol]:

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
    ) -> None:
        ...

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
    ) -> RetrievalOutputType:
        ...
```

We're automatically wrapping `Encoder` and `CrossEncoder` models support `SearchProtocol`. However, if your model needs a custom index you can implement this protocol directly, like was done for colbert-like models.

#### New Documentation

We've added a lot of new documentation to make it easier to get started with MTEB.

- You can see api of our models in tasks in [API documentation](./api/index.md).
- We've added a [getting started guide](./usage/get_started.md) to help you get started with MTEB.
- You can see implemented [tasks](./overview/available_tasks/retrieval.md) and [models](../overview/available_models/text.md) in MTEB.

#### Better support for loading and comparing results

The new `ResultCache` also makes it easier to load, inspect and compare both local and online results:

```py
from mteb.cache import ResultCache

cache = ResultCache(cache_path="~/.cache/mteb") # default
cache.download_from_remote() # download the latest results from the remote repository

# load both local and online results
results = cache.load_results(models=["sentence-transformers/all-MiniLM-L6-v2", ...], tasks=["STS12"])
df = results.to_dataframe()
```

#### Descriptive Statistics

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

Documentation for [the descriptive statistics types][mteb.types.statistics].

#### Saving Predictions

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

#### Support datasets v4

With the new functionality for [reuploading datasets][mteb.abstasks.abstask.AbsTask.push_dataset_to_hub] to the standard datasets Parquet format, we’ve reuploaded all tasks with `trust_remote_code`, and MTEB now fully supports Datasets v4.

### Upgrading from v1

This section gives an introduction of how to upgrade from v1 to v2.

#### Replacing `mteb.MTEB`

The previous approach to evaluate would require you to first create `MTEB` object and then call `.run` on that object.
The `MTEB` object was initially a sort of catch all object intended for both filtering tasks, selecting tasks, evaluating and few other cases.

This overload of functionality made it hard to change. We have already for a while made it easier to filter and select tasks using `get_tasks` and
`mteb.evaluate` now superseded `MTEB` as the method for evaluation.

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

#### Replacing `mteb.load_results()`

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


#### Converting model to new format

As mentioned in [the above section](#multimodal-input-format) MTEB v2, now supports multimodal input as the default.
Luckily for you all models implemented in MTEB already supports this new format! However, if you have a local model that you would like to evaluate
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

#### Reuploading datasets

If your dataset is in old format, or you want to reupload it to the new Parquet format, you can do so using the new
[`push_dataset_to_hub`][mteb.abstasks.abstask.AbsTask.push_dataset_to_hub] method:

```py
import mteb

task = mteb.get_task("MyOldTask")
task.push_dataset_to_hub("my-username/my-new-task")
```

#### Converting Reranking datasets to new format

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
