"""Model metadata for Scooby Embed (Funk&Love).

Place at ``mteb/models/model_implementations/funkandlove_models.py``.

scooby-embed-v1 starts from thenlper/gte-small-zh (embedding rows unreachable by its
tokenizer removed, all 6 layers kept) and is distilled from richinfoai/ritrieve_zh_v1.
``training_datasets`` therefore includes the teacher's declared training data plus the
MTEB tasks whose train splits (texts only) appear in our own training corpus:
classification and STS train splits, CLUE CMNLI/OCNLI train, and the train queries and
passage collections of T2Ranking, DuReader-retrieval and mMARCO (Chinese).
No evaluation split was used; texts identical to any non-train split of MTEB(cmn, v1)
were removed from the training corpus.
"""

from __future__ import annotations

from mteb.models.model_implementations.richinfoai_models import ritrieve_zh_v1
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

SCOOBY_TRAINING_DATASETS = {
    # classification train splits
    "TNews",
    "IFlyTek",
    "Waimai",
    "OnlineShopping",
    "JDReview",
    "MultilingualSentiment",
    # STS / pair train splits
    "LCQMC",
    "AFQMC",
    "BQ",
    "ATEC",
    "PAWSX",
    "STSB",
    # CLUE CMNLI / OCNLI train
    "Cmnli",
    "Ocnli",
    # train queries and passage collections
    "T2Retrieval",
    "T2Reranking",
    "DuRetrieval",
    "MMarcoRetrieval",
    "MMarcoReranking",
}

scooby_embed_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="funkandlove/scooby-embed-v1",
    model_type=["dense"],
    languages=["zho-Hans"],
    open_weights=True,
    revision="cc068b8f98f1b283c1bb6c839f9e2508058e00ea",
    release_date="2026-11-25",
    n_parameters=26_998_272,
    n_embedding_parameters=7_557_120,  # token 14,760 x 512 (position embeddings excluded)
    memory_usage_mb=103,  # fp32 safetensors
    max_tokens=512,
    embed_dim=512,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/funkandlove/scooby-embed-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=SCOOBY_TRAINING_DATASETS | ritrieve_zh_v1.training_datasets,
    adapted_from="thenlper/gte-small-zh",
    superseded_by=None,
    contacts=["Hofmann8"],
    citation=None,
)
