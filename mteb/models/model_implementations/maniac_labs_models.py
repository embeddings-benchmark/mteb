"""
MTEB ModelMeta for ManiacLabs/miniac-embed.

LEAF-distilled embedding model: E5-small-unsupervised backbone, mxbai-embed-large teacher.
"""

from mteb.models import ModelMeta, sentence_transformers_loader

MINIAC_EMBED_TRAINING_DATASETS = {
    "fineweb",
    "cc_news",
    "english-words-definitions",
    "amazon-qa",
    "MSMARCO",
    "PubMedQA",
    "trivia_qa",
    "LoTTE",
}

miniac_embed = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(),
    name="ManiacLabs/miniac-embed",
    model_type=["dense"],
    revision="0fe5413163ce75cf13e6351b39a8b6f321b64e79",
    release_date="2026-02-13",
    languages=["eng-Latn"],
    open_weights=True,
    framework=[
        "PyTorch",
        "Sentence Transformers",
    ],
    n_parameters=33_360_000,
    n_embedding_parameters=11_917_056,
    memory_usage_mb=127,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/ManiacLabs/miniac-embed",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="intfloat/e5-small-unsupervised",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=MINIAC_EMBED_TRAINING_DATASETS,
    citation=None,
)
