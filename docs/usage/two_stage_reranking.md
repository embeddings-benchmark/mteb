# Two stage reranking

The two-stage reranking approach is a powerful technique used in information retrieval and natural language processing to improve the quality of search results or recommendations. It involves two main stages: an initial retrieval stage and a subsequent reranking stage.

The first step will be to retrieve a broad set of candidate items using an encoder and efficient method.

```python
import mteb

encoder_model = mteb.get_model("minishlab/potion-base-2M")
task = mteb.get_task("NanoArguAnaRetrieval")
prediction_folder = "prediction_folder"

results = mteb.evaluate(
    encoder_model,
    task,
    prediction_folder=prediction_folder,
)
```

The second step will be to rerank the top-k candidates using a cross-encoder model. You can specify the `top_k` parameter to control how many candidates to rerank.

```python
task = task.convert_to_reranking(task.predictions_path(prediction_folder), top_k=100)
cross_encoder_model = mteb.get_model("jinaai/jina-reranker-v2-base-multilingual")
results = mteb.evaluate(
    cross_encoder_model,
    task,
)
```
