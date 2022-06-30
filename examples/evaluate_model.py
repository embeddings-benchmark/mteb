import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)


model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)
evaluation = MTEB(task_langs=["en"])
evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"])

print("--DONE--")
