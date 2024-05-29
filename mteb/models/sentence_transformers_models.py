"""Implementation of Sentence Transformers model validated in MTEB."""

from mteb.model_meta import ModelMeta

ALL_MINILM_L6_V2 = ModelMeta(
    name="sentence-transformers/all-MiniLM-L6-v2",
    languages=["eng-Latn"],
    open_source=True,
    revision="e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
    release_date="2021-08-30",
)
