from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.arctic_models import arctic_v1_training_datasets
from mteb.models.mxbai_models import mixedbread_training_data

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}

LEAF_TRAINING_DATASETS = {
    "AmazonQA": ["train"],
    "LoTTE": ["dev", "test"],
    # FineWeb
    # CC-News
    # PubMedQA
    # TriviaQA
}

mdbr_leaf_ir = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="MongoDB/mdbr-leaf-ir",
        revision="2e46f5aac796e621d51f678c306a66ede4712ecb",
        model_prompts=model_prompts,
    ),
    name="MongoDB/mdbr-leaf-ir",
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
    reference="https://huggingface.co/MongoDB/mdbr-leaf-ir",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={**LEAF_TRAINING_DATASETS, **arctic_v1_training_datasets},
)

mdbr_leaf_mt = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="MongoDB/mdbr-leaf-mt",
        revision="66c47ba6d753efc208d54412b5af6c744a39a4df",
        model_prompts=model_prompts,
    ),
    name="MongoDB/mdbr-leaf-mt",
    revision="66c47ba6d753efc208d54412b5af6c744a39a4df",
    release_date="2025-08-27",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    n_parameters=22_958_592,
    memory_usage_mb=86,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/MongoDB/mdbr-leaf-mt",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={**LEAF_TRAINING_DATASETS, **mixedbread_training_data},
)
