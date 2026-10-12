---
title: "Two stage reranking"
icon: lucide/list-ordered
---


## Two stage reranking

To use a cross encoder for reranking on a retrieval task you first need a task, a stage 1 model and our cross-encoder.

```python
import mteb

task = mteb.get_task("NanoArguAnaRetrieval")
# stage 1 model:
encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
# stage 2 model:
cross_encoder = mteb.get_model("cross-encoder/ms-marco-TinyBERT-L2-v2")  # (1)!
```

1.  You can also directly use `CrossEncoder` from [sentence transformers](https://www.sbert.net/).

Once we have that we can perform stage 1 retrieval, followed by a stage 2 reranking (call [`convert_to_reranking`][mteb.abstasks.retrieval.AbsTaskRetrieval.convert_to_reranking] to convert the task to a reranking task, which will use the predictions from stage 1 as input for stage 2):

```python
prediction_folder = "model_predictions"

# stage 1: retrieval
res = mteb.evaluate(
    encoder,
    task,
    prediction_folder=prediction_folder,
)

# convert task to retrieval
task = task.convert_to_reranking(prediction_folder, top_k=100)

# stage 2: reranking
cross_enc_results = mteb.evaluate(cross_encoder, task)

print(task.metadata.main_score)  # NDCG@10
res[0].get_score()  # 0.286
cross_enc_results[0].get_score()  # 0.338
```

## Selecting document modalities

Use [`SearchCrossEncoderWrapper`][mteb.models.SearchCrossEncoderWrapper] to select
which document inputs a reranker receives:

```python
from mteb.models import SearchCrossEncoderWrapper

# Reuse the cross_encoder and converted task from the example above.
text_reranker = SearchCrossEncoderWrapper(
    cross_encoder, document_modalities=["text"]
)
results = mteb.evaluate(text_reranker, task)
```

You can also pass the selection when loading a registered reranker:

```python
text_reranker = mteb.get_model(
    "cross-encoder/ms-marco-TinyBERT-L2-v2", document_modalities=["text"]
)
```

For a multimodal corpus and a compatible reranker, use `["image"]` for document
images or `["text", "image"]` for both. Each requested modality must be present in
the task corpus and supported by the model. Query inputs, candidate documents and
relevance labels are unchanged. Text selection includes document titles and bodies;
image-only selection removes those text fields. Without `document_modalities`,
the wrapper keeps the existing task inputs and cache behavior.

Explicit selections are saved in `model_meta.json`, for example:

```json
{"experiment_kwargs": {"document_modalities": ["image", "text"]}}
```

The list is sorted and deduplicated, and other experiment settings are preserved.
The normal result cache gives each configuration a separate directory:

```text
results/<reranker>/<revision>/experiments/<experiment_id>/
    model_meta.json
    <task>.json
```

This records the document modalities supplied to the reranker, separately from its
supported `modalities`. It lets a leaderboard group configurations under the same
model and filter by document inputs without overwriting another run. Omitted
selections remain unspecified; their modality cannot be inferred from capabilities.
Leaderboard grouping is separate from this input-selection feature.
