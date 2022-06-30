import logging

from mteb import MTEB
from mteb.tasks.BitextMining import BUCCBitextMining
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
        BUCCBitextMining(langs=["de-en"]),
        "STS12",
        "SummEval",
    ]
)
eval.run(model)
