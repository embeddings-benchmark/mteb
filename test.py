from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.Reranking import MindSmallReranking

model = SentenceTransformer("average_word_embeddings_komninos")
eval = MTEB(tasks=["TwitterURLCorpus", "TwitterSemEval2015", "SprintDuplicateQuestions"])
eval.run(model)
