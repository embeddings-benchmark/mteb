---
title: "Compress embeddings"
icon: lucide/minimize-2
---

## Compressing Embeddings

Storing dense embeddings for later re-use, i.e. as an index to enable retrieval, requires significant amounts of storage, especially for large corpora. An easy way to alleviate this issue is to compress the embeddings by representing each embedding dimension with a lower number of bits, i.e. going from 32 bits to 8, which will consume four times less storage. While reducing storage consumption, however, embedding compression can lead to reduced performance. You can evaluate the performance of compressed embeddings for a model by using the [`CompressionWrapper`][mteb.models.compression_wrappers.compression_wrapper.CompressionWrapper] class:

```python
import mteb
from mteb.models import CompressionWrapper
from mteb.types import OutputDType

# define your task(s) and model above as normal
task = mteb.get_task("AILACasedocs")
model = mteb.get_model("intfloat/multilingual-e5-large-instruct")

# wrap the model with the compression wrapper
model_with_compression = CompressionWrapper(model, output_dtype=OutputDType.INT8)
# run as normal
results = mteb.evaluate(model_with_compression, tasks=[task])
```

The `output_dtype` parameter determines the value range to which embeddings are compressed. Consult the [`OutputDType`][mteb.types.OutputDType] class for a full overview of valid compression levels.

### Threshold Estimation

#### How it works

When compressing to integer output types, i.e. int8, floating point numbers are mapped to discrete integer values. This requires estimating thresholds to determine which range of floats is mapped to a specific integer. For this, the wrapper calculates the minimum and maximum values per embedding dimension, then divides the value range in between into 2^x equal parts, with x being the number of bits to compress to, i.e. 8.

#### Clipping embeddings

Naturally, the process described above is prone to outliers, i.e. maximum and minimum values that rarely occur and are far outside the usual value range. As a result, the integer distribution of the compressed embeddings can be highly imbalanced. To ensure a better value distribution, the top and bottom x percentile of embedding values can be clipped:

```python
import mteb
from mteb.models import CompressionWrapper
from mteb.types import OutputDType

model = mteb.get_model("intfloat/multilingual-e5-large-instruct")
model_with_compression = CompressionWrapper(model, output_dtype=OutputDType.INT8, clipping_margin=(0.025, 0.975))
```
