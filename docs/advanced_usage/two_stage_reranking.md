## Two stage reranking

To use a cross encoder for reranking. The following code shows a two-stage run with the second stage reading results saved from the first stage.

```python
from sentence_transformers import CrossEncoder

import mteb

encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
task = mteb.get_task("NanoArguAnaRetrieval")

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
# if model implemented in mteb it's better to use `mteb.get_model`
# cross_encoder = mteb.get_model("jinaai/jina-reranker-v2-base-multilingual")
# or if model is't implemented you can pass CrossEncoder directly
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
cross_enc_results = mteb.evaluate(cross_encoder, task)

print(task.metadata.main_score) # NDCG@10
res[0].get_score()  # 0.286
cross_enc_results[0].get_score() # 0.338
```
