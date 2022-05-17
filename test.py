from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.kNNClassification import MassiveIntentClassification

model = SentenceTransformer("average_word_embeddings_komninos") 
task = MassiveIntentClassification(langs=["en", "de"])
task = "MassiveIntentClassification"
task = "Banking77Classification"
eval = MTEB(task_list=[task])
# print(eval.available_tasks)
eval.run(model)