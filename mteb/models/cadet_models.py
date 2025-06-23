from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.bge_models import bge_m3_training_data

cadet_training_data = {
    # we train with the corpora of FEVER, MSMARCO, and DBPEDIA. We only train with synthetic generated queries.
    # However, we do use queries from MSMARCO as examples for synthetic query generation.
    "MSMARCO": ["train"],
    "DBPedia": [],
    # We distill from RankT5 which trains using MSMARCO + NQ.
    # source: https://arxiv.org/pdf/2210.10634
    "NQ": ["train"],
    # We also distill from bge-rerankerv2.5-gemma2-lightweight, which utilizes the BGE-M3 dataset for training, along with Arguana, HotpotQA, and FEVER.
    # source: https://arxiv.org/pdf/2409.15700
    "FEVER": ["train"],
    "HotpotQA": ["train"],
    "ArguAna": ["train"],
}

for k, v in bge_m3_training_data.items():
    cadet_training_data.setdefault(k, []).extend(v)
# deduplicate
cadet_training_data = {
    k: list(dict.fromkeys(v)) for k, v in cadet_training_data.items()
}


cadet_embed = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="manveertamber/cadet-embed-base-v1",
        revision="8056d118be37a566f20972a5f35cda815f6bc47e",
        model_prompts={
            "query": "query: ",
            "passage": "passage: ",
        },
    ),
    name="manveertamber/cadet-embed-base-v1",
    languages=["eng-Latn"],
    revision="8056d118be37a566f20972a5f35cda815f6bc47e",
    open_weights=True,
    release_date="2025-05-11",
    n_parameters=109_000_000,
    memory_usage_mb=418,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/manveertamber/cadet-embed-base-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/manveertamber/cadet-dense-retrieval",
    # we provide the code to generate the training data
    public_training_data="https://github.com/manveertamber/cadet-dense-retrieval",
    training_datasets=cadet_training_data,
    adapted_from="intfloat/e5-base-unsupervised",
)
