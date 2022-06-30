import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)


model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)
evaluation = MTEB(task_types=["Clustering"])
evaluation.run(model, output_folder=f"results/{model_name}")

print("--DONE--")
