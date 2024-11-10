from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

import mteb

params = []

# add all model names
model_names = [  # "google/siglip-base-patch16-512",
    "google/siglip-so400m-patch14-384"
    # ...
]

for model_name in tqdm(model_names):
    model = mteb.get_model(model_name)

    total_params = sum(p.numel() for p in model.model.parameters())
    total_params = total_params / 1e6
    params.append([model_name, total_params])

    del model
    torch.cuda.empty_cache()

param_frame = pd.DataFrame(params, columns=["model name", "# params"])
param_frame.to_csv("params.csv", index=False)
