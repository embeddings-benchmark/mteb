import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.tasks.BitextMining import BUCCBitextMining

logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(
    tasks=[
        "Banking77Classification",
        "TwentyNewsgroupsClustering",
        "SciDocsRR",
        "SprintDuplicateQuestions",
        "NFCorpus",
        BUCCBitextMining(langs=["de-en"]),
        "STS12",
        "SummEval",
    ]
)
eval.run(model)
