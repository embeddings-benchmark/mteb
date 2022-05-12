from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.kNNClassification import MassiveIntentClassification

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(task_list=[MassiveIntentClassification(["en"]), "MassiveIntentClassification"])
print(eval.selected_tasks)
eval.run(model)
