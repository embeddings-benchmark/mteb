from functools import partial

from mteb.model_meta import ModelMeta
from mteb.loaders.sentence_transformers import sentence_transformers_loader

my_model = ModelMeta(
    name="OrdalieTech/Solon-embeddings-mini-beta-1.1",
    languages=["fra-Latn"],
    open_weights=True,
    revision="8e4ea66eb7eb6109b47b7d97d7556f154d9aec4a",
    release_date="2025-01-01",
    embed_dim=768,  # ← ajuste si différent
    license="apache-2.0",
    max_tokens=8192,  # ← ajuste si besoin
    reference="https://huggingface.co/OrdalieTech/Solon-embeddings-mini-beta-1.1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    # Données d'entraînement (non-MTEB) + privé synthétique LLM
    training_data=(
        "https://huggingface.co/datasets/PleIAs/common_corpus; "
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb; "
        "https://huggingface.co/datasets/OrdalieTech/wiki_fr; "
        "private LLM-synthetic (train)"
    ),
    # Loader explicite ST
    loader=partial(
        sentence_transformers_loader,
        model_name="OrdalieTech/Solon-embeddings-mini-beta-1.1",
        revision="8e4ea66eb7eb6109b47b7d97d7556f154d9aec4a",
        trust_remote_code=True,
    ),
)
