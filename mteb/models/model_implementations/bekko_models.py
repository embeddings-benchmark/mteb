from typing import Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper


class BekkoEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Sentence Transformers loader with Bekko's documented L2 normalization."""

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["normalize_embeddings"] = True
        return super().encode(*args, **kwargs)


BEKKO_TRAINING_DATASETS = {
    "HotpotQA",
    "MIRACLRetrievalHardNegatives",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NQ",
    "NanoMSMARCORetrieval",
    "PawsXPairClassification",
}

BEKKO_COMMON_KWARGS = {
    "loader": BekkoEncoderWrapper,
    "loader_kwargs": {
        "model_kwargs": {
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
        }
    },
    "languages": None,
    "n_embedding_parameters": 98_304_000,
    "max_tokens": 8192,
    "embed_dim": [64, 128, 256, 384],
    "license": "mit",
    "open_weights": True,
    "public_training_code": None,
    "public_training_data": (
        "https://huggingface.co/datasets/hotchpotch/bekko-embedding-v1-unsupervised"
    ),
    "framework": [
        "Sentence Transformers",
        "PyTorch",
        "safetensors",
        "Transformers",
        "ONNX",
        "OpenVINO",
    ],
    "similarity_fn_name": ScoringFunction.COSINE,
    "use_instructions": False,
    "training_datasets": BEKKO_TRAINING_DATASETS,
    "adapted_from": "jhu-clsp/mmBERT-small",
    "modalities": ["text"],
    "model_type": ["dense"],
    "citation": """@misc{tateno2026bekko,
  title={Bekko Embedding: Parameter-Efficient Multilingual Retrieval with Ultra-Compact Encoders},
  author={Yuichi Tateno},
  year={2026},
  eprint={2607.25180},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2607.25180},
}""",
    "contacts": ["hotchpotch"],
}


bekko_embedding_v1_a8m = ModelMeta(
    name="hotchpotch/bekko-embedding-v1-a8m",
    revision="a8cedb6b46fad5df6f10e1c94750ff62e298fbd2",
    n_parameters=105_975_168,
    memory_usage_mb=202,
    release_date="2026-07-28",
    reference="https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m",
    **BEKKO_COMMON_KWARGS,
)


bekko_embedding_v1_a25m = ModelMeta(
    name="hotchpotch/bekko-embedding-v1-a25m",
    revision="aede922b27c1751d091b1e26c86dc82446a46cb4",
    n_parameters=123_234_432,
    memory_usage_mb=470,
    release_date="2026-07-28",
    reference="https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m",
    **BEKKO_COMMON_KWARGS,
)
