from __future__ import annotations
from functools import partial
from mteb import ModelMeta
from sentence_transformers import SentenceTransformer
import torch

# Define task instructions with specific task names
def searchmap_loader():
    model = SentenceTransformer(
        "VPLabs/SearchMap_Preview", 
        trust_remote_code=True
    )

    if hasattr(model, "encoder"):
        # Configure for inference only
        model.encoder.eval()  # Ensure eval mode
        torch.set_grad_enabled(False)  # Disable gradients
    return model

searchmap_preview = ModelMeta(
    loader=searchmap_loader,
    name="VPLabs/SearchMap_Preview",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    revision="69de17ef48278ed08ba1a4e65ead8179912b696e",
    release_date="2025-03-05",
    n_parameters=435_000_000,
    memory_usage_mb=1660,
    embed_dim=4096,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/VPLabs/SearchMap_Preview",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code=None,
    public_training_data=None,
    training_datasets=None
) 