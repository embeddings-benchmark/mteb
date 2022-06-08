from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.Reranking import MindSmallReranking

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(tasks=['Touche2020', 'SCIDOCS', 'SciFact'])
print(eval.selected_tasks)

eval.run(model)