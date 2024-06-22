from mteb import MTEB
from mteb.tasks.Retrieval.eng.BrightRetrieval import BrightRetrieval
from sentence_transformers import SentenceTransformer

# testing the task with a model:
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
evaluation = MTEB(tasks=[BrightRetrieval()])
evaluation.run(model, output_folder=f"results")
