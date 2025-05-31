from __future__ import annotations

from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.tasks.Retrieval.eng.R2MEDRetrieval import R2MEDRetrieval

# testing the task with a model:
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
evaluation = MTEB(tasks=[R2MEDRetrieval()])
evaluation.run(model, output_folder="results")
