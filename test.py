from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.kNNClassification import MassiveIntentClassification

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(task_list=[MassiveIntentClassification(["de", "123"])])
# print(eval.available_tasks)
eval.run(model)