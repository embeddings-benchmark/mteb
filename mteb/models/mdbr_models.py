from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.arctic_models import arctic_v1_training_datasets
from mteb.models.mxbai_models import mixedbread_training_data

mdb_leaf_ir = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="MongoDB/mdb-leaf-ir",
        revision="2e46f5aac796e621d51f678c306a66ede4712ecb",
    ),
    name="MongoDB/mdb-leaf-ir",
    revision="2e46f5aac796e621d51f678c306a66ede4712ecb",
    release_date="2025-08-27",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=22_861_056,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/MongoDB/mdb-leaf-ir",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=arctic_v1_training_datasets,
)

mdb_leaf_mt = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="MongoDB/mdb-leaf-mt",
        revision="1",
    ),
    name="MongoDB/mdb-leaf-mt",
    revision="1",
    release_date="2025-08-27",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=22_958_592,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/MongoDB/mdb-leaf-mt",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=mixedbread_training_data,
)
