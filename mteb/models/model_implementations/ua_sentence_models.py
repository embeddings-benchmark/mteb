"""Sentence models for evaluation on the Ukrainian part of MTEB"""

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

xlm_roberta_ua_distilled = ModelMeta(
    name="panalexeu/xlm-roberta-ua-distilled",
    loader=sentence_transformers_loader,
    n_parameters=278_000_000,
    memory_usage_mb=1061,
    max_tokens=512,
    embed_dim=768,
    revision="9216f50d76b032350ca312246fa2f5dcaa6ca971",
    release_date="2025-04-15",
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/panalexeu/xlm-roberta-ua-distilled/blob/main/researches/research_final.ipynb",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://github.com/panalexeu/xlm-roberta-ua-distilled/tree/main",
    languages=["eng-Latn", "ukr-Cyrl"],
    training_datasets=set(
        #  "sentence-transformers/parallel-sentences-talks",
        #  "sentence-transformers/parallel-sentences-wikimatrix",
        #  "sentence-transformers/parallel-sentences-tatoeba",
    ),
    adapted_from="FacebookAI/xlm-roberta-base",
    modalities=["text"],
    public_training_data=None,
    use_instructions=False,
)
