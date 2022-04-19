from sentence_transformers import SentenceTransformer
from mteb.evaluation import *

model = SentenceTransformer('average_word_embeddings_komninos')
eval = MTEB(task_list=['TwitterURLCorpus','TwitterSemEval2015','SprintDuplicateQuestions'])
print(eval.available_tasks)
print(eval.selected_tasks)
eval.run(model)