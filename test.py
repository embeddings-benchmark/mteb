from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.Reranking import MindSmallReranking

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(task_types=["Retrieval", "Classification"])
eval.mteb_tasks()
eval.selected_tasks()