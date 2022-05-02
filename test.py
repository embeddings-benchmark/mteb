from sentence_transformers import SentenceTransformer
from mteb import MTEB

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB(task_types=['Reranking'])
eval.run(model)