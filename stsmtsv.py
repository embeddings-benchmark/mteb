import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"
# or directly from huggingface:
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["STSMTSV"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")