from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.tasks.Retrieval.eng.BrightRetrieval import BrightRetrieval

# testing the task with a model:
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
evaluation = MTEB(tasks=[BrightRetrieval()])
evaluation.run(model, output_folder="results")
