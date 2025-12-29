import time

import mteb

"""
model_name = "BAAI/bge-m3"
model = mteb.get_model(model_name)
tasks = mteb.get_tasks(tasks=["T2Reranking"])
start = time.perf_counter()
# When using batch size is 12, the gpu memory usage is 18.669 GB, and it is estimated to take 13 hours to complete.
results = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 12, "show_progress_bar": True})
end = time.perf_counter()
elapsed_time = end - start
main_score = results[0].scores["dev"][0]["main_score"]
print("elapsed_time", elapsed_time, "main_score", main_score)
"""


model_name = "BAAI/bge-m3"
model = mteb.get_model(model_name, config_kwargs={"dtype": "float16"})
tasks = mteb.get_tasks(tasks=["T2Reranking"])
start = time.perf_counter()
# There is not much speed improvement even when using float16. it is estimated to take 12 hours to complete.
results = mteb.evaluate(
    model, tasks=tasks, encode_kwargs={"batch_size": 12, "show_progress_bar": True}
)
end = time.perf_counter()
elapsed_time = end - start
main_score = results[0].scores["dev"][0]["main_score"]
print("elapsed_time", elapsed_time, "main_score", main_score)
