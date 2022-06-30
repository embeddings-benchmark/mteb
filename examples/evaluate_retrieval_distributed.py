import logging
import time

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    model = SentenceTransformer("msmarco-distilbert-base-tas-b")

    eval = MTEB(tasks=["NFCorpus"])
    start_time = time.time()
    eval.run(model, corpus_chunk_size=50000, batch_size=256, eval_splits=["test"], output_folder=None)

    print()
    print("--- %s seconds ---" % (round(time.time() - start_time, 2)))
    print()
