from __future__ import annotations

import json
from pathlib import Path

import mteb

path = Path("/home/tmp")
model_name_folders = [
    f
    for f in path.iterdir()
    if f.is_dir() and "linear" not in f.name and "." not in f.name
]
revision_folders = [
    sf
    for folder in model_name_folders
    for sf in folder.iterdir()
    if sf.is_dir() and "." not in sf.name
]
model_names = [f.name for f in model_name_folders]
revisions = [f.name for f in revision_folders]

models = []
for m in model_names:
    if "EVA" in m:
        models.append(f"QuanSun/{m}")
    elif "voyage" in m:
        models.append(f"voyageai/{m}")
    else:
        models.append(m.replace("__", "/"))


base_results_path = Path("/home/results/results")
for m, r in zip(models, revisions):
    print(m)
    mm = mteb.get_model_meta(model_name=m, revision=r)
    print(mm.to_dict())

    target_path = base_results_path / m.replace("/", "__") / r / "model_meta.json"

    with open(target_path, "w") as f:
        json.dump(mm.to_dict(), f)
