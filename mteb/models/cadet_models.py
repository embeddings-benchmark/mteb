from __future__ import annotations

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader
from functools import partial

cadet_training_data = {
    "MSMARCO": ["train"],
    # we train with the corpora of FEVER, MSMARCO, and DBPEDIA. We only train with synthetic generated queries (but we do use query examples from the MSMARCO train set), so perhaps it is not suitable to list "train" for FEVER and DBPEDIA?
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
    public_training_data="https://github.com/manveertamber/cadet-dense-retrieval", # we provide the code to generate the training data
    training_datasets=cadet_training_data,
    adapted_from="intfloat/e5-base-unsupervised",
)

