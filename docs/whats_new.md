## New in v2.0 🎉
<!-- TODO: not finished -->
This section goes through new features added in v2. Below we give an overview of changes following by detailed examples.

**Overview of changes:**

- Easier evaluation using `mteb.evaluate`
- Easier caching and results loading using the `ResultCache`
- New documentation
- Standardization of file names and typing across the library 

What are the reason for the changes? Generally the many inconsistencies in the library made it hard to maintain without introducing breaking changes and we do think that there is multiple import areas to expand in, e.g. [adding new benchamrk for image embeddings](https://arxiv.org/abs/2504.10471), support new model types in general making the library more accecible. 
We have already been able to add many new feature in v2.0, but hope that this new version allow us to keep doing so without breaking backward compatibility. See [upgrading from v1](#upgrading-from-v1) for specific deprecations and the reasoning behind them.


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

<!-- MORE NEW FEATURES -->



## Upgrading from v1

This section gives an introduction of how to upgrade from v1 to v2.


### Replacing `MTEB`

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
