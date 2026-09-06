from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

OCTEN_LAW_8B_V1_CITATION = "@misc{octen-law-8b-v1,\n  title={Octen Law 8B v1: a multilingual legal text embedding model},\n  author={{Litil Labs}},\n  year={2026},\n  howpublished={\\url{https://huggingface.co/litillabs/octen-law-8b-v1}}\n}"


litillabs_octen_law_8b_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="litillabs/octen-law-8b-v1",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    open_weights=True,
    revision="cc2b41645060edebf7246cb8b53064173a03b6c4",
    release_date="2026-08-25",
    n_parameters=7_567_295_488,
    n_embedding_parameters=621_219_840,
    memory_usage_mb=14_433,
    embed_dim=4096,
    max_tokens=40_960,
    license="apache-2.0",
    reference="https://huggingface.co/litillabs/octen-law-8b-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    # WikiQA TRAIN was used directly. GerDaLIRSmall and LeCaRDv2 are marked
    # conservatively because the training pack contains disclosed
    # cross-direction benchmark-adjacent near-duplicates for those tasks.
    training_datasets={"GerDaLIRSmall", "LeCaRDv2", "WikiQA"},
    citation=OCTEN_LAW_8B_V1_CITATION,
    adapted_from="Octen/Octen-Embedding-8B",
)
