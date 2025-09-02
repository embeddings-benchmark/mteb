# Usage

This usage documentation follows a structure similar first it introduces a simple example of how to evaluate a model in MTEB.
Then introduces model detailed section of defining model, selecting tasks and running the evaluation. Each section contains subsections pertaining to
these.


## Evaluating a Model

Evaluating a model on MTEB follows a three step approach, 1) defining model, 2) selecting the tasks and 3) running the evaluation

```python
import mteb

# Specify the model that we want to evaluate
model = ...

# specify what you want to evaluate it on
tasks = mteb.get_tasks(tasks=["{task1}", "{task1}"])

# run the evaluation
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```

For instance if we want to run [`"sentence-transformers/all-MiniLM-L6-v2"`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on
`"Banking77Classification"` we can do this using the following code:

```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"


# load the model using MTEB
model = mteb.get_model(model_name) # will default to SentenceTransformers(model_name) if not implemented in MTEB
# or using SentenceTransformers
model = SentenceTransformers(model_name)

# select the desired tasks and evaluate
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```


## Speeding up evaluations

Evaluation in MTEB consists of three main components. The download of the dataset, the encoding of the samples, and the evaluation. Typically, the most notable bottleneck are either in the encoding step or on the download step. We discuss how to speed these up in the following sections.

### Speeding up download

The fastest way to speed up downloads is by using Huggingface's [`xet`](https://huggingface.co/blog/xet-on-the-hub). You can use this simply using:

```bash
pip install mteb[xet]
```

For one of the larger datasets, `MrTidyRetrieval` (~15 GB), we have seen speed-ups from ~40 minutes to ~30 minutes while using `xet`.

### Evaluating on Different Modalities
MTEB is not only text evaluating, but also allow you to evaluate image and image-text embeddings.

> [!NOTE]
> Running MTEB on images requires you to install the optional dependencies using `pip install mteb[image]`

To evaluate image embeddings you can follow the same approach for any other task in `mteb`. Simply ensuring that the task contains the modality "image":

```python
tasks = mteb.get_tasks(modalities=["image"]) # Only select tasks with image modalities
task = task[0]

print(task.metadata.modalites)
# ['text', 'image']
```

However, we recommend starting with one of the predefined benchmarks:

```python
import mteb
benchmark = mteb.get_benchmark("MIEB(eng)")
evaluation = mteb.MTEB(tasks=benchmark)

model = mteb.get_model("{model-of-choice}")
evaluation.run(model)
```

You can also specify exclusive modality filtering to only get tasks with exactly the requested modalities (default behavior with `exclusive_modality_filter=False`):
```python
# Get tasks with image modality, this will also include tasks having both text and image modalities
tasks = mteb.get_tasks(modalities=["image"], exclusive_modality_filter=False)

# Get tasks that have ONLY image modality
tasks = mteb.get_tasks(modalities=["image"], exclusive_modality_filter=True)
```






## Defining a Model

### Using a pre-defined Model

MTEB comes with an implementation of many popular models and APIs. These can be loaded using `mteb.get_model_meta`:

```python
model_name = "intfloat/multilingual-e5-small"
meta = mteb.get_model_meta(model_name)
model = meta.load_model()
# or directly using
model = mteb.get_model(model_name)
```

You can get an overview of the models available in `mteb` as follows:

```py
model_metas = mteb.get_model_metas()

# You can e.g. use the model metas to find all openai models
openai_models = [meta for meta in model_metas if "openai" in meta.name]
```
> [!TIP]
> Some models require additional dependencies to run on MTEB. An example of such a model is the OpenAI APIs.
> These dependencies can be installed using `pip install mteb[openai]`

### Using a Sentence Transformer Model

MTEB is made to be compatible with sentence transformers and thus you can readily evaluate any model that can be loaded via. sentence transformers
on `MTEB`:

```python
model = SentenceTransformers("sentence-transformers/LaBSE")

# select the desired tasks and evaluate
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```

However, we do recommend checking if mteb includes an implementation of the model before using sentence transformers since some models (e.g. the [multilingual e5 models](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings-67b2b8bb9bff40dec9fb3534)) require a prompt and not specifying it may reduce performance.

> [!NOTE]
> If you want to evaluate a cross encoder on a reranking task, see section on [running cross encoders for reranking](#running-cross-encoders-on-reranking)

### Using a Custom Model

It is also possible to implement your own custom model in MTEB as long as it adheres to the [encoder interface](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/encoder_interface.py#L21).

This entails implementing an `encode` function taking as input a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).

```python
import mteb
from mteb.encoder_interface import PromptType
import numpy as np


class CustomModel:
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        pass


# evaluating the model:
model = CustomModel()
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model)
```

If you want to submit your implementation to be included in the leaderboard see the section on [submitting a model](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md).

## Selecting Tasks

This section describes how to select benchmarks and tasks to evaluate, including selecting specific subsets or splits to run.

### Selecting a Benchmark

`mteb` comes with a set of predefined benchmarks. These can be fetched using `mteb.get_benchmark` and run in a similar fashion to other sets of tasks.
For instance to select the English datasets that form the English leaderboard:

```python
import mteb
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
evaluation = mteb.MTEB(tasks=benchmark)
```

The benchmark specified not only a list of tasks, but also what splits and language to run on.

To get an overview of all available benchmarks, simply run:

```python
import mteb
benchmarks = mteb.get_benchmarks()
```

> [!NOTE]
> Generally we use the naming scheme for benchmarks `MTEB(*)`, where the "*" denotes the target of the benchmark.
> In the case of a language, we use the three-letter language code.
> For large groups of languages, we use the group notation, e.g., `MTEB(Scandinavian, v1)` for Scandinavian languages.
> External benchmarks implemented in MTEB like `CoIR` use their original name.

When using a benchmark from MTEB please cite `mteb` along with the citations of the benchmark which you can access using:

```python
benchmark.citation
```

### Task selection

`mteb` comes with the utility function `mteb.get_task` and `mteb_get_tasks` for fetching and analysing the tasks of interest.

This can be done in multiple ways, e.g.:

* by the task name
* by their type (e.g. "Clustering" or "Classification")
* by their languages
* by their domains
* by their modalities
* and many more

```python
# by name
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
# by type
tasks = mteb.get_tasks(task_types=["Clustering", "Retrieval"]) # Only select clustering and retrieval tasks
# by language
tasks = mteb.get_tasks(languages=["eng", "deu"]) # Only select datasets which contain "eng" or "deu" (iso 639-3 codes)
# by domain
tasks = get_tasks(domains=["Legal"])
# by modality
tasks = mteb.get_tasks(modalities=["text", "image"]) # Only select tasks with text or image modalities
# or using multiple
tasks = get_tasks(languages=["eng", "deu"], script=["Latn"], domains=["Legal"])
```

For more information see the documention for `mteb.get_tasks`

You can also specify which languages to load for multilingual/cross-lingual tasks like below:

```python
import mteb

tasks = [
    mteb.get_task("AmazonReviewsClassification", languages = ["eng", "fra"]),
    mteb.get_task("BUCCBitextMining", languages = ["deu"]), # all subsets containing "deu"
]
```

### Selecting Evaluation Split or Subsets
A task in `mteb` mirrors the structure of a dataset on Huggingface. It includes a splits (i.e. "test") and a subset.

```python
# selecting an evaluation split
task = mteb.get_task("Banking77Classification", eval_splits=["test"])
# selecting a Huggingface subset
task = mteb.get_task("AmazonReviewsClassification", hf_subsets=["en", "fr"])
```

> [!NOTE]
>  **What is a subset?** A subset on a Huggingface dataset is what you specify after the dataset name, e.g. `datasets.load_dataset("nyu-mll/glue", "cola")`.
> Often the subset does not need to be defined and is left as "default". The subset is however useful, especially for multilingual datasets to specify the
> desired language or language pair e.g. in [`mteb/bucc-bitext-mining`](https://huggingface.co/datasets/mteb/bucc-bitext-mining) we might want to evaluate only on the French-English subset `"fr-en"`.




### Using a Custom Task

To evaluate on a custom task, you can run the following code on your custom task.
See [how to add a new task](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_dataset.md), for how to create a new task in MTEB.


```python
import mteb
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking


class MyCustomTask(AbsTaskReranking):
    ...

model = mteb.get_model(...)
evaluation = mteb.MTEB(tasks=[MyCustomTask()])
evaluation.run(model)
```


## Running the Evaluation

This section contains documentation related to the runtime of the evaluation. How to pass arguments to the encoder, saving outputs and similar.


### Introduction to the runner

By default `mteb` will save the results in the `results/{model_name}` folder, however if you want to save the results in a specific folder you
can specify it as follows:

```python
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="my_results_folder")
```

### Tracking Carbon Emissions

`mteb` allows for easy tracking of carbon emission eq. using `codecarbon`. You simply need to install `mteb[codecarbon]` and enable co2 tracking:

```python
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, co2_tracker=True)
```


### Passing in `encode` arguments

To pass in arguments to the model's `encode` function, you can use the encode keyword arguments (`encode_kwargs`):

```python
evaluation.run(model, encode_kwargs={"batch_size": 32})
```

### Running SentenceTransformer model with prompts

Prompts can be passed to the SentenceTransformer model using the `prompts` parameter. The following code shows how to use prompts with SentenceTransformer:

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("average_word_embeddings_komninos", prompts={"query": "Query:", "passage": "Passage:"})
evaluation = mteb.MTEB(tasks=tasks)
```

In prompts the key can be:
1. Prompt types (`document`, `query`) - they will be used in reranking and retrieval tasks
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


### Running Cross Encoders on Reranking

To use a cross encoder for reranking, you can directly use a CrossEncoder from SentenceTransformers. The following code shows a two-stage run with the second stage reading results saved from the first stage.

```python
from mteb import MTEB
import mteb
from sentence_transformers import CrossEncoder, SentenceTransformer

cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
dual_encoder = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

subset = "default" # subset name used in the NFCorpus dataset
eval_splits = ["test"]

evaluation = MTEB(tasks=tasks)
evaluation.run(
    dual_encoder,
    eval_splits=eval_splits,
    save_predictions=True,
    output_folder="results/stage1",
)
evaluation.run(
    cross_encoder,
    eval_splits=eval_splits,
    top_k=5,
    save_predictions=True,
    output_folder="results/stage2",
    previous_results=f"results/stage1/NFCorpus_{subset}_predictions.json",
)
```


### Using Late Interaction Models

This section outlines how to use late interaction models for retrieval.

```python
from mteb import MTEB
import mteb


colbert = mteb.get_model("colbert-ir/colbertv2.0")
tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

eval_splits = ["test"]

evaluation = MTEB(tasks=tasks)

evaluation.run(
    colbert,
    eval_splits=eval_splits,
    corpus_chunk_size=500,
)
```
This implementation employs the MaxSim operation to compute the similarity between sentences. While MaxSim provides high-quality results, it processes a larger number of embeddings, potentially leading to increased resource usage. To manage resource consumption, consider lowering the `corpus_chunk_size` parameter.


### Saving retrieval task predictions

To save the predictions from a retrieval task, add the `--save_predictions` flag in the CLI or set `save_predictions=True` in the run method. The filename will be in the "{task_name}_{subset}_predictions.json" format.

Python:
```python
from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

evaluation = MTEB(tasks=tasks)
evaluation.run(
    model,
    eval_splits=["test"],
    save_predictions=True,
    output_folder="results",
)
```

CLI:
```bash
mteb run -t NFCorpus -m all-MiniLM-L6-v2 --output_folder results --save_predictions
```

### Caching Embeddings To Re-Use Them

There are times you may want to cache the embeddings so you can re-use them. This may be true if you have multiple query sets for the same corpus (e.g. Wikipedia) or are doing some optimization over the queries (e.g. prompting, other experiments). You can setup a cache by using a simple wrapper, which will save the cache per task in the `<path_to_cache_dir>/<task_name>` folder:

```python
# define your task(s) and model above as normal
task = mteb.get_task("LccSentimentClassification")
model = mteb.get_model("minishlab/M2V_base_glove_subword")
evaluation = mteb.MTEB(tasks=[task])

# wrap the model with the cache wrapper
from mteb.models.cache_wrapper import CachedEmbeddingWrapper
model_with_cached_emb = CachedEmbeddingWrapper(model, cache_path='path_to_cache_dir')
# run as normal
evaluation.run(model_with_cached_emb)
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

## Leaderboard

This section contains information on how to interact with the leaderboard including running it locally, analysing the results, annotating contamination and more.

### Annotate Contamination

have your found contamination in the training data of a model? Please let us know, either by opening an issue or ideally by submitting a PR
annotating the training datasets of the model:

```py
model_w_contamination = ModelMeta(
    name = "model-with-contamination"
    ...
    training_datasets: {"ArguAna": # name of dataset within MTEB
                        ["test"]} # the splits that have been trained on
    ...
)
```


### Running the Leaderboard Locally

It is possible to completely deploy the leaderboard locally or self-host it. This can e.g. be relevant for companies that might want to
integrate build their own benchmarks or integrate custom tasks into existing benchmarks.

Running the leaderboard is quite easy. Simply run:
```bash
make run-leaderboard
```

The leaderboard requires gradio install, which can be installed using `pip install mteb[leaderboard]` and requires python >3.10.
