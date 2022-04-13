from sentence_transformers import SentenceTransformer
from mteb.evaluation import *

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB(tasks_list=['RedditClustering', 'QuoraRetrieval'])
eval.run(model)