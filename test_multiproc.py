import logging
import time

from mteb import MTEB
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    model = SentenceTransformer("average_word_embeddings_komninos")
    # model = SentenceTransformer("msmarco-distilbert-base-tas-b")

    eval = MTEB(tasks=["SprintDuplicateQuestions"])
    # eval = MTEB(tasks=["FiQA2018"])
    # eval = MTEB(tasks=["MSMARCO"])
    start_time = time.time()
    # eval.run(model, corpus_chunk_size=4500, output_folder=None, target_devices=["cpu"] * 2)
    eval.run(model, corpus_chunk_size=256, batch_size=256, output_folder=None, eval_splits=["test"])

    print()
    print("--- %s seconds ---" % (round(time.time() - start_time, 2)))
    print()
