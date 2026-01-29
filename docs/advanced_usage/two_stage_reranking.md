## Two stage reranking

To use a cross encoder for reranking on a retrieval task you first need a task, an stage 1 model and our cross-encoder.

```python
import mteb

task = mteb.get_task("NanoArguAnaRetrieval")
# stage 1 model:
encoder = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
# stage 2 model:
cross_encoder = mteb.get_model("cross-encoder/ms-marco-TinyBERT-L-2-v2") # (1)
```

1.  You can also directly use `CrossEncoder` from [sentence transformers](https://www.sbert.net/).

Once we have that we we can the perform stage 1 retrieval, followed by a stage 2 reranking:

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

print(task.metadata.main_score) # NDCG@10
res[0].get_score()  # 0.286
cross_enc_results[0].get_score() # 0.338
```