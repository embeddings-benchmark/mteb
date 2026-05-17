---
title: "Define the Model"
icon: lucide/bot
---

# Defining the Model

## Using a pre-defined Model

MTEB comes with an implementation of many popular models and APIs. These can be loaded using [`mteb.get_model_meta`](../../api/model.md#mteb.get_model_meta) or [`mteb.get_model`](../../api/model.md#mteb.get_model):

```python
model_name = "intfloat/multilingual-e5-small"
meta = mteb.get_model_meta(model_name) # (1)!
model = meta.load_model()
# or directly using
model = mteb.get_model(model_name)
```

1.  Using `mteb.get_model_meta` allows us to work with the model without loading it. E.g. you can pass it to `mteb.evaluate`, which only loads the model if the results doesn't already exist.

You can get an overview of the models available in `mteb` as follows:

```python
model_metas = mteb.get_model_metas()

# You can e.g. use the model metas to find all openai models
openai_models = [meta for meta in model_metas if "openai" in meta.name]
```

!!! tip
    Some models require additional dependencies to run on MTEB. An example of such a model is the OpenAI APIs.
    These dependencies can be installed using `pip install mteb[openai]` or `uv add "mteb[openai]"`

## Using a Sentence Transformer Model

MTEB is made to be compatible with sentence transformers and thus you can readily evaluate any model that can be loaded via. [sentence transformers](https://www.sbert.net/).
on `MTEB`:

=== "SentenceTransformers"

    ```python
    import mteb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformers("sentence-transformers/LaBSE")

    # select the desired tasks and evaluate
    tasks = mteb.get_tasks(tasks=["Banking77Classification"])
    results = mteb.evaluate(model, tasks=tasks)
    ```

=== "CrossEncoder"

    ```python
    import mteb
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("sentence-transformers/LaBSE")

    # select a reranking task and evaluate
    tasks = mteb.get_tasks(tasks=["AskUbuntuDupQuestions"])
    results = mteb.evaluate(model, tasks=tasks)
    ```



!!! note
    We do recommend using `mteb.get_model` which will by default load the model using the implementation in `mteb` if there is one, otherwise it will use `SentenceTransformers` or `CrossEncoder` from [sentence transformers](https://www.sbert.net/) if appropriate. The `mteb` implementations typically differ due to models requiring specific prompts or similar hyperparameters, and not specifying these may reduce performance (e.g. the [multilingual e5 models](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings-67b2b8bb9bff40dec9fb3534) require specific prompts).

## Using a Custom Model

It is also possible to implement your own custom model in MTEB as long as it adheres to the [EncoderProtocol][mteb.models.EncoderProtocol].

This entails implementing an `encode` function taking as input a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).

```python
import mteb
from mteb.types import PromptType, Array
import numpy as np


class CustomModel:
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> Array:
        """Encodes the given sentences using the encoder.

        Args:
            inputs: The inputs to encode.
            task_metadata: The name of the task.
            hf_subset: The subset of the dataset.
            hf_split: The split of the dataset.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        pass


# evaluating the model:
model = CustomModel()
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
model = mteb.evaluate(model, tasks=tasks)
```

If you want to submit your implementation to be included in the leaderboard see the section on [submitting a model](../../contributing/adding_a_model.md).

## Using BM25 Baselines

MTEB includes language-aware BM25 baselines that can be loaded like any other model:

```python
import mteb

# Language-aware BM25: auto-selects stopwords, stemmer, and tokenizer from task metadata
model = mteb.get_model("mteb/baseline-bm25s")

# Subword BM25: uses a HuggingFace subword tokenizer (Qwen3) for better multilingual coverage
model = mteb.get_model("mteb/baseline-bm25s-subword")
```

??? info "Performance comparison on Chinese retrieval"

    The tokenizer choice has a large impact for non-Latin scripts. Results on [LeCaRDv2](https://huggingface.co/datasets/mteb/LeCaRDv2) (Chinese legal case retrieval, 3 795 docs, 159 queries):

    | Model / tokenizer | ndcg@10 |
    |---|---|
    | `mteb/baseline-bm25s` (mteb<=2.13.5) | 0.359 |
    | `mteb/baseline-bm25s` (mteb>2.13.6) | 0.567 |
    | `mteb/baseline-bm25s-subword` (Qwen3-0.6B) | 0.631 |
    | Custom Jieba tokenizer (see example below) | 0.641 |

    We recommend using a bm25s-subword as a language agnostic baseline, as it performs reasonably on non-latin as well as latin languages,
    but if you know the language you can often obtain better performance using a language-specific tokenizer. The differences in the mteb implementation
    stems for PR [4405](https://github.com/embeddings-benchmark/mteb/pull/4405) which either uses a white-space or character level tokenization depending on the language.
    For Chinese it is character level.  

### Custom tokenizer

You can pass any `text -> list[str]` callable as a custom tokenizer, or provide a HuggingFace tokenizer name:

```python
import mteb

# Using a HuggingFace tokenizer by name (e.g. for a specific language)
model = mteb.get_model("mteb/baseline-bm25s", tokenizer="bert-base-multilingual-cased")

# Using a custom callable (e.g. Jieba for Chinese)
import jieba

def jieba_tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]

model = mteb.get_model("mteb/baseline-bm25s", tokenizer=jieba_tokenize)
```
