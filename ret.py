import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
# model_name = "average_word_embeddings_komninos"
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["STSIDMT"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["STSIDMT"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")