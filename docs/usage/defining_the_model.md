# Defining the Model

## Using a pre-defined Model

MTEB comes with an implementation of many popular models and APIs. These can be loaded using [`mteb.get_model_meta`](../api/model.md#mteb.get_model_meta) or [`mteb.get_model`](../api/model.md#mteb.get_model):

```python
model_name = "intfloat/multilingual-e5-small"
meta = mteb.get_model_meta(model_name)
model = meta.load_model()
# or directly using
model = mteb.get_model(model_name)
```

You can get an overview of the models available in `mteb` as follows:

```python
model_metas = mteb.get_model_metas()

# You can e.g. use the model metas to find all openai models
openai_models = [meta for meta in model_metas if "openai" in meta.name]
```

!!! tip
    Some models require additional dependencies to run on MTEB. An example of such a model is the OpenAI APIs.
    These dependencies can be installed using `pip install mteb[openai]`

## Using a Sentence Transformer Model

MTEB is made to be compatible with sentence transformers and thus you can readily evaluate any model that can be loaded via. sentence transformers
on `MTEB`:

```python
model = SentenceTransformers("sentence-transformers/LaBSE")

# select the desired tasks and evaluate
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
results = mteb.evaluate(model, tasks=tasks)
```

However, we do recommend checking if mteb includes an implementation of the model before using sentence transformers since some models (e.g. the [multilingual e5 models](https://huggingface.co/collections/intfloat/multilingual-e5-text-embeddings-67b2b8bb9bff40dec9fb3534)) require a prompt and not specifying it may reduce performance.

!!! note
    If you want to evaluate a cross encoder for reranking, see the section on [running cross encoders for reranking](./running_the_evaluation.md#running-cross-encoders-on-reranking).

## Using a Custom Model

It is also possible to implement your own custom model in MTEB as long as it adheres to the [EncoderProtocol][mteb.models.EncoderProtocol].

This entails implementing an `encode` function taking as input a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).

```python
import mteb
from mteb.types import PromptType
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
    ) -> np.ndarray:
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

If you want to submit your implementation to be included in the leaderboard see the section on [submitting a model](../contributing/adding_a_model.md).
