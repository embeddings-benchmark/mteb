from sentence_transformers import SentenceTransformer
from mteb.evaluation import *

model = SentenceTransformer('all-MiniLM-L6-v2')
eval = MTEB()
eval.run(model)