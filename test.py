from sentence_transformers import SentenceTransformer
from mteb.evaluation import *

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB()
eval.run(model)