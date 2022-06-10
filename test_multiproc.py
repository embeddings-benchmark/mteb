from sentence_transformers import SentenceTransformer
from mteb import MTEB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # model = SentenceTransformer("average_word_embeddings_komninos")
    model = SentenceTransformer("msmarco-distilbert-base-tas-b")

    # eval = MTEB(tasks=["ArguAna"])
    eval = MTEB(tasks=["FiQA2018"])
    eval.run(model, corpus_chunk_size=None, output_folder="results3", target_devices=["cpu"] * 2)
