from sentence_transformers import SentenceTransformer
from mteb.evaluation import *

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB(task_list=['RedditClustering', 'QuoraRetrieval'])
print(eval.available_task_categories)
print(eval.available_task_types)
print(eval.available_tasks)
print(eval.selected_tasks)
eval.run(model)