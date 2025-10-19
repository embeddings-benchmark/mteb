## Adding a model to the Leaderboard

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard). To submit to it:

1. Add the [model meta](#adding-a-model-implementation) to `mteb`
2. [Evaluate](../usage/get_started.md#evaluating-a-model) the desired model using `mteb` on the [benchmarks](../usage/selecting_tasks.md#selecting-a-benchmark)
3. Push the results to the [results repository](https://github.com/embeddings-benchmark/results) via a PR. Once merged they will appear on the leaderboard after a day.


## Adding a model implementation

Adding a model implementation to `mteb` is quite straightforward.
Typically it only requires that you fill in metadata about the model and add it to the [model directory](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations):

??? example "Adding a ModelMeta object"
    ```python
    from mteb.models import ModelMeta, sentence_transformers_loader

    my_model = ModelMeta(
        name="model_name",
        loader=sentence_transformers_loader,
        languages=["eng-Latn"], # follows ISO 639-3 and BCP-47
        open_weights=True,
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        release_date="2025-01-01",
        n_parameters=568_000_000,
        memory_usage_mb=2167,
        embed_dim=4096,
        license="mit",
        max_tokens=8194,
        reference="https://huggingface.co/user-or-org/model-name",
        similarity_fn_name="cosine",
        framework=["Sentence Transformers", "PyTorch"],
        use_instructions=False,
        public_training_code="https://github.com/user-or-org/my-training-code",
        public_training_data="https://huggingface.co/datasets/user-or-org/full-dataset",
        training_datasets={"MSMARCO"}, # if you trained on the MSMARCO training set
    )
    ```

This works for all [Sentence Transformers](https://sbert.net) compatible models. Once filled out, you can submit your model to `mteb` by submitting a PR.


### Calculating the Memory Usage

To calculate `memory_usage_mb`, run:

```py
model_meta = mteb.get_model_meta("model_name")
model_meta.calculate_memory_usage_mb()
```

### Adding instruction models

Some models, such as the [E5 models][@wang2024multilingual], use instructions or prefixes.
You can directly add the prompts when saving and uploading your model to the Hub. Refer to this [configuration file as an example](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/blob/3b5a16eaf17e47bd997da998988dce5877a57092/config_sentence_transformers.json).

However, you can also add these directly to the model configuration:

```python
model = ModelMeta(
    loader=sentence_transformers_loader
    loader_kwargs=dict(
        model_prompts={
           "query": "query: ",
           "passage": "passage: ",
        },
    ),
    ...
)
```

### Using a custom Implementation

If you need to use a custom implementation, you can specify the `loader` parameter in the [`ModelMeta`][mteb.models.model_meta.ModelMeta] class. It should implement one of the following protocols: [Encoder][mteb.models.EncoderProtocol], [CrossEncoder][mteb.models.CrossEncoderProtocol], or [Search][mteb.models.SearchProtocol].

??? example "Custom Model Implementation"
    ```python
    from mteb.types import PromptType
    import numpy as np

    class CustomModel:
        def __init__(self, model_name: str, revision: str, **kwargs):
            pass # your initialization of model here

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            prompt_type: PromptType | None = None,
            **kwargs,
        ) -> np.ndarray:

            arrays = []
            for batch in inputs:
                documents = batch["text"]
                # embed documents:
                embed_dim = 100
                embedding = np.zeros((len(documents), embed_dim))

            embeddings = np.concat(arrays)
            return embeddings
    ```

Then you can specify the `loader` parameter in the `ModelMeta` class:

```python
your_model = ModelMeta(
    loader=CustomModel,
    loader_kwargs={...},
    ...
)
```


### Adding model dependencies
If you are adding a model that requires additional dependencies, you can add them to the `pyproject.toml` file, under optional dependencies:

```toml
voyageai = ["voyageai>=1.0.0,<2.0.0"]
```

This ensures that the implementation does not break if a package is updated.

As it is an optional dependency, you can't use top-level dependencies, but will instead have to use import inside the wrapper scope:

??? example "Adding optional dependencies"
    ```python
    from mteb._requires_package import requires_package

    class VoyageAIModel:
        def __init__(self, model_name: str, revision: str, **kwargs) -> None:
            requires_package(self, "voyageai", model_name, "pip install 'mteb[voyageai]'")
            import voyageai
            ...
    ```

### Submitting your model as a PR

When submitting you models as a PR, please copy and paste the following checklist into pull request message:

```markdown
- [ ] I have filled out the ModelMeta object to the extent possible
- [ ] I have ensured that my model can be loaded using
  - [ ] `mteb.get_model(model_name, revision)` and
  - [ ] `mteb.get_model_meta(model_name, revision)`
- [ ] I have tested the implementation works on a representative set of tasks.
- [ ] The model is public, i.e. is available either as an API or the weight are publicly available to download
```
