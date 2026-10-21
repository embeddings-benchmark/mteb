"""Bekko Static Embedding with the standard Sentence Transformers loader."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

bekko_static_embedding_v1 = ModelMeta(
    name="hotchpotch/bekko-static-embedding-v1",
    revision="c18167b8f244c1bc6de21ce9b710bac8a29c4615",
    loader=SentenceTransformerEncoderWrapper,
    languages=[
        "eng-Latn",
        "arb-Arab",
        "bul-Cyrl",
        "ben-Beng",
        "cat-Latn",
        "ces-Latn",
        "dan-Latn",
        "deu-Latn",
        "ell-Grek",
        "spa-Latn",
        "ekk-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "heb-Hebr",
        "hin-Deva",
        "hrv-Latn",
        "hun-Latn",
        "ind-Latn",
        "ita-Latn",
        "jpn-Jpan",
        "kor-Hang",
        "nld-Latn",
        "nob-Latn",
        "nno-Latn",
        "pol-Latn",
        "por-Latn",
        "ron-Latn",
        "rus-Cyrl",
        "slk-Latn",
        "slv-Latn",
        "srp-Cyrl",
        "srp-Latn",
        "swe-Latn",
        "swh-Latn",
        "tel-Telu",
        "tha-Thai",
        "tur-Latn",
        "ukr-Cyrl",
        "vie-Latn",
        "yor-Latn",
        "cmn-Hani",
    ],
    release_date="2026-10-21",
    n_parameters=204_800_000,
    n_embedding_parameters=204_800_000,
    memory_usage_mb=391,
    max_tokens=32768,
    embed_dim=[32, 64, 128, 256, 512, 1024],
    license="mit",
    open_weights=True,
    public_training_code=(
        "https://huggingface.co/hotchpotch/bekko-static-embedding-v1/"
        "tree/c18167b8f244c1bc6de21ce9b710bac8a29c4615/training_scripts"
    ),
    public_training_data=(
        "https://huggingface.co/datasets/hotchpotch/bekko-embedding-v1-unsupervised"
    ),
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/hotchpotch/bekko-static-embedding-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    # Conservative task-family disclosures for MIRACL and all-nli training data.
    # Further corpus exposure (DBpedia, AG News, Quora, StackExchange, etc.) is
    # described in the submission notes; this is not an exhaustive overlap audit.
    training_datasets={"MIRACLRetrieval", "MIRACLRetrievalHardNegatives", "XNLI"},
    adapted_from=None,
    modalities=["text"],
    model_type=["dense"],
    contacts=["hotchpotch"],
)
