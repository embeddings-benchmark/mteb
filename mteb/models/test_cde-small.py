from __future__ import annotations

import mteb

model_name = "jxm/cde-small-v1"
revision = "main"

model = mteb.get_model(
    "jxm/cde-small-v1",
    trust_remote_code=True,
    model_prompts={"query": "search_query: ", "passage": "search_document: "},
)
print("succesffully got the model!")

print(mteb.get_model_meta(model_name, revision))

