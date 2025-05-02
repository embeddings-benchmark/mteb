from __future__ import annotations

import mteb

model_name = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
revision = "45e42c89990c40aca042659133fc8b13c28634b5"
# model = mteb.get_model(model_name=model_name, revision=revision)
model = mteb.get_model_meta(model_name, revision)
print(f"Model {model_name} loaded successfully.")
