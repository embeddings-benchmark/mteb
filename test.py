from sentence_transformers import SentenceTransformer
from mteb import MTEB

# set logging INFO
import logging

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # model = SentenceTransformer("average_word_embeddings_komninos")
    model = SentenceTransformer("msmarco-distilbert-base-tas-b")

    eval = MTEB(tasks=["MSMARCOv2"])
    eval.run(model, corpus_chunk_size=50000)
