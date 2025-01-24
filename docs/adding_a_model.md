## Adding a Model to the MTEB Leaderboard

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard). To submit to it:

1. Add meta information about your model to [model dir](../mteb/models/).
   ```python
   from mteb.model_meta import ModelMeta
    
   bge_m3 = ModelMeta(
       name="model_name",
       languages=["model_languages"], # in format eng-Latn
       open_weights=True,
       revision="5617a9f61b028005a4858fdac845db406aefb181",
       release_date="2024-06-28",
       n_parameters=568_000_000,
       embed_dim=4096,
       license="mit",
       max_tokens=8194,
       reference="https://huggingface.co/BAAI/bge-m3",
       similarity_fn_name="cosine",
       framework=["Sentence Transformers", "PyTorch"],
       use_instructions=False,
       public_training_code=None,
       public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
       training_datasets={"your_dataset": ["train"]},
   )
   ```
   By default model will run using the [`sentence_transformers_loader`](../mteb/models/sentence_transformer_wrapper.py) function. If you need to use a custom implementation, you can specify the `loader` parameter in the `ModelMeta` class. For example:
   ```python
   from mteb.models.wrapper import Wrapper
   from mteb.encoder_interface import PromptType
   import numpy as np
   
   class CustomWrapper(Wrapper):
       def __init__(self, model_name, model_revision):
           super().__init__(model_name, model_revision)
           # your custom implementation here
       
       def encode(
            self,
            sentences: list[str],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs
       ) -> np.ndarray:
           # your custom implementation here
           return np.zeros((len(sentences), self.embed_dim))
   ```
   Then you can specify the `loader` parameter in the `ModelMeta` class:
   ```python
   your_model = ModelMeta(
       loader=partial(
            CustomWrapper, 
            model_name="model_name",
            model_revision="5617a9f61b028005a4858fdac845db406aefb181"
       ),
       ...
   )
   ```
2. **Run the desired model on MTEB:**

Either use the Python API:

```python
import mteb

# load a model from the hub (or for a custom implementation see https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md)
model = mteb.get_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

tasks = mteb.get_tasks(...) # get specific tasks
# or 
tasks = mteb.get_benchmark("MTEB(eng, classic)") # or use a specific benchmark

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model, output_folder="results")
```

Or using the command line interface:

```bash
mteb run -m {model_name} -t {task_names}
```

These will save the results in a folder called `results/{model_name}/{model_revision}`.

2. **Push Results to the Leaderboard**

To add results to the public leaderboard you can push your results to the [results repository](https://github.com/embeddings-benchmark/results) via a PR. Once merged they will appear on the leaderboard after a day.

4. **Wait for a refresh the leaderboard**

**Notes:**

##### Using Prompts with Sentence Transformers

If your model uses Sentence Transformers and requires different prompts for encoding the queries and corpus, you can take advantage of the `prompts` [parameter](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer). 

Internally, `mteb` uses the prompt named `query` for encoding the queries and `passage` as the prompt name for encoding the corpus. This is aligned with the default names used by Sentence Transformers.

###### Adding the prompts in the model configuration (Preferred)

You can directly add the prompts when saving and uploading your model to the Hub. For an example, refer to this [configuration file](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/blob/3b5a16eaf17e47bd997da998988dce5877a57092/config_sentence_transformers.json).

```python
model = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="intfloat/multilingual-e5-small",
        revision="fd1525a9fd15316a2d503bf26ab031a61d056e98",
        model_prompts={
           "query": "query: ",
           "passage": "passage: ",
        },
    ),
)
```
