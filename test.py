from sentence_transformers import SentenceTransformer
from mteb import MTEB

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB(task_list=['StackExchangeClusteringP2P'])
print(eval.selected_tasks)
eval.run(model)