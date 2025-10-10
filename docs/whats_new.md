# New in v2.0 🎉
This section goes through new features added in v2. Below we give an overview of changes following by detailed examples.

**Overview of changes:**

- Easier evaluation using `mteb.evaluate`
- Easier caching and results loading using the `ResultCache`
- Support for multimodal evaluation
- New documentation
- Descriptive statistics for all tasks
- Better support for error analysis, allowing you to save predictions 
- Standardization of file names and typing across the library
- Consistent logging and progress bars
- And much more

What are the reason for the changes? Generally the many inconsistencies in the library made it hard to maintain without introducing breaking changes and we do think that there is multiple import areas to expand in, e.g. [adding new benchamrk for image embeddings](https://arxiv.org/abs/2504.10471), support new model types in general making the library more accecible. 
We have already been able to add many new feature in v2.0, but hope that this new version allow us to keep doing so without breaking backward compatibility. See [upgrading from v1](#upgrading-from-v1) for specific deprecations and how to fix them.


### Easier evalation

Evaluations is not a lot easier using `mteb.evaluate`, 

```py
results = mteb.evaluate(model, tasks)
```

### Better local and online caching
The new `mteb.ResultCache` makes managing the cache notably easier:
```py
from mteb.cache import ResultCache

model = ...
tasks = ...

cache = ResultCache(cache_path="~/.cache/mteb") # default

# simple evaluate with cache
results = mteb.evaluate(model, tasks, cache=cache) # only runs if results not in cache
```

It allow you to access the online cache so you don't have to rerun existing models.

```py
# no need to rerun already public results
cache.download_from_remote() # download the latest results from the remote repository
results = mteb.evaluate(model, tasks, cache=cache)
```

### Better support for loading and comparing results

The new `ResultCache` also makes it easier to load, inspect and compare both local and online results:

```py
from mteb.cache import ResultCache

cache = ResultCache(cache_path="~/.cache/mteb") # default
cache.download_from_remote() # download the latest results from the remote repository

# load both local and online results
results = cache.load_results(models=["sentence-transformers/all-MiniLM-L6-v2", ...], tasks=["STS12"])
df = results.to_dataframe()
```

### Multimodal Input format

Models in mteb who implements the [`Encoder`](api/model.md#mteb.models.encoder) protocol now supports multimodal input With the model protocol roughly looking like so:

```py
class Encoder(Protocol): # simplified
    """The interface for an encoder in MTEB."""

    def encode(self, inputs: DataLoader[BatchedInput], ...) -> Array: ...
```

Not only does this allow more efficient loading using the torch dataloader, but it also allows keys for multiple modalities:

```py
batch_input: BatchedInput = {
    text: list[str],
    images: list[list[images]],
    audio: list[list[audio]], # upcoming
    # + optional fields such as document title
}
```

Where `text` is a batch of texts and `list[images]` is a batch for that texts. This e.g. allows markdown documents with multiple figures like so: 

> As you see in the following figure [figure 1](image_1) there is a correlation between A and B. This is similarly seen in figure 2 [figure 2](image_2)

However this also allows no text, multi-image inputs (e.g. for PDFs). Overall this greatly expands the possible tasks that can now be evaluated in MTEB.
To see how to convert a legacy model see the [converting model](#converting-model-to-new-format) section.

### Descriptive Statistics

Descriptive statistics isn't a new thing in MTEB, however, now it is there for every task, to extract it simply run:

```py
import mteb
task = mteb.get_task("MIRACLRetrievalHardNegatives")

task.metadata.descriptive_stats
```

And you will get a highly detailed set of descriptive statistics covering everything from number of samples query lengths, duplicates, etc.. These not only make it easier for you to examine tasks but it also make it easier for us to make quality checks on future tasks.

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


## Upgrading from v1

This section gives an introduction of how to upgrade from v1 to v2.


### Replacing `mteb.MTEB`

The previous approach to evaluate would require you to first create `MTEB` object and then call `.run` on that object. 
The `MTEB` object was initially a sort of catch all object intended for both filtering tasks, selecting tasks, evaluating and few other cases.

This overload of functionality made it hard to change. We have already for a while made it easier to filter and select tasks using `get_tasks` and
`mteb.evaluate` now superseeded `MTEB` as the method for evaluation. 

```py
# Approach before 2.0.0:
eval = mteb.MTEB(tasks=tasks) # now throw a deprecation warning
results = eval.run(model, 
    overwrite=True,
    encode_kwargs={},
    ...
)

# Recommended:
mteb.evaluate(model, tasks, 
    overwrite_strategy="only-missing" # only rerun missing splits
    encode_kwargs = {},
    ...
    )
```

### Replacing `mteb.load_results()`

Given the new `ResultCache` makes dealing with a results from _both_ local and online caches a lot easier, it can now replace `mteb.load_results` it 

```py
tasks = mteb.get_tasks(tasks=["STS12"])
model_names = ["intfloat/multilingual-e5-large"]

# Approach before 2.0.0:
results = mteb.load_results(models = model_names, tasks=tasks, download_latest=True)

# Recommended:
cache = ResultCache("~/.cache/mteb") # default
cache.download_from_remote() # downloads remote results

results = cache.load_results(models=model_names, tasks=tasks)
```


### Converting model to new format

As mentioned in [the above section](#multimodal-input-format) MTEB v2, now supports multimodal input as the default. 
Luckily for you all models implemented in MTEB already supports this new format! However, if you have a local model that you would like to evaluate
Here is a quick conversion guide. If you previous implementation looks like so:

```py
# v1.X.X
class MyDummyEncoder:
    def __init__(self, **kwargs):
        self.embed_dim = 10

    def encode(self, sentences: list[str], **kwargs) -> Array:
        embeddings = np.random.rand(len(sentences), self.embed_dim)
        return embeddings
```

You can simply unpack it to its text input like so:

```py
# v2.0.0
class MyDummyEncoder:
    def __init__(self, **kwargs):
        self.embed_dim = 10

    def encode(self, input: DataLoader[BatchedInput], **kwargs) -> Array:
        # unpack to v1 format:
        sentences = [text for batch in inputs for text in batch["text"]]
        # do as you did beforehand:
        embeddings = np.random.rand(len(sentences), self.embed_dim)
        return embeddings
```

Of course it will be more efficient if work directly with the dataloader.


