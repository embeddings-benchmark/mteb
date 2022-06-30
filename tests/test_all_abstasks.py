import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(
    tasks=[
        "Banking77Classification",
        "TwentyNewsgroupsClustering",
        "SciDocs",
        "SprintDuplicateQuestions",
        "NFCorpus",
        "BUCC",
        "STS12",
        "SummEval",
    ]
)
eval.run(model)
