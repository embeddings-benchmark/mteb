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

### Different Cache backends

By default, the `CachedEmbeddingWrapper` uses a NumPy memmap backend (`NumpyCache`) to store embeddings. However, you can also use other backends. Currently, only `FAISS` is implemented, but you can provide your own custom backend that implements the `CacheBackendProtocol` by passing it as the `search_backend` parameter when initializing the `CachedEmbeddingWrapper`. For example:

```python
import mteb
from mteb.models.cache_wrappers.cache_backends import FaissCache
from mteb.models import CachedEmbeddingWrapper

model = mteb.get_model(...)
cachedmodel = CachedEmbeddingWrapper(model, "cache_dir", cache_backend=FaissCache)
```
