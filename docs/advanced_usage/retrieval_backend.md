# Retrieval Search backend

!!! note "Available since 2.3.0"
    This feature was introduced in version **2.3.0**.

For some large dataset search can take a lot of time and memory. To reduce this you can use `FaissSearchIndex`. To work with it install `pip install mteb[faiss]`.

Usage example:
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

For example running `minishlab/potion-base-2M` on `SWEbenchVerifiedRR` took 694 seconds instead of 769.
