from mteb.evaluation import MTEB
from mteb import get_tasks, get_model

model_name = "facebook/wav2vec2-base-960h"
model = get_model(model_name=model_name)

tasks = get_tasks(tasks=["CIFAR10ZeroShot"])
evaluation = MTEB(tasks=tasks)
results = evaluation.run(model)