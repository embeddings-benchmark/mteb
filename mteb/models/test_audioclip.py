from __future__ import annotations

from mteb import get_model, get_tasks
from mteb.evaluation import MTEB

model_name = "laion/clap-htsat-unfused"
model = get_model(model_name)

tasks = get_tasks(tasks=["ESC50ZeroShot"])
evaluation = MTEB(tasks=tasks)
results = evaluation.run(model)
print(results[-1].to_dict())