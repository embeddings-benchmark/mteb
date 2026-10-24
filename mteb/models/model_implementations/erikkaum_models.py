"""Model definitions for Erik Kaum's embedding models."""

import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

# The Stage 2 training data contains FiQA, NQ, HotpotQA, MS MARCO, FEVER,
# SQuAD v2, and TriviaQA. SQuAD v2 and TriviaQA do not have direct MTEB task
# counterparts. Nano task variants are derived from their corresponding source
# datasets and are therefore also marked as training-data overlaps.
_LATTICE_TRAINING_DATASETS = {
    "FEVER",
    "FEVERHardNegatives",
    "FiQA2018",
    "HotpotQA",
    "HotpotQAHardNegatives",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NQ",
    "NQHardNegatives",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNQRetrieval",
}


lattice_retrieval = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="erikkaum/lattice-retrieval",
    revision="0925206b12cddc454956c2fb3207a0285bf12701",
    release_date="2026-08-06",
    languages=["eng-Latn"],
    n_parameters=31_254_528,
    n_embedding_parameters=31_254_528,
    memory_usage_mb=119,
    max_tokens=np.inf,
    embed_dim=[32, 64, 128, 256, 512, 1024],
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/ErikKaum/lattice/tree/main/trainer",
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/erikkaum/lattice-retrieval",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_LATTICE_TRAINING_DATASETS,
    modalities=["text"],
    model_type=["dense"],
    contacts=["ErikKaum"],
    citation=None,
)
