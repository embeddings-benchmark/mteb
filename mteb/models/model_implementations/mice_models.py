from __future__ import annotations

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

mice_p_citation = """
@misc{morand2026micemeetyoureranking,
  title={MICE to Meet You: From Re-ranking Only to End-to-End Retrieval},
  author={Victor Morand and Yuxuan Zong and Mathias Vast and Josiane Mothe and Benjamin Piwowarski},
  year={2026},
  eprint={2610.31932},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2610.31932}
}
"""

_MICE_P_COMMON = dict(
    loader=CrossEncoderWrapper,
    loader_kwargs={"trust_remote_code": True},
    model_type=["cross-encoder"],
    languages=["eng-Latn"],
    open_weights=True,
    release_date="2026-10-15",
    max_tokens=7999,
    embed_dim=None,
    license="apache-2.0",
    similarity_fn_name=None,
    use_instructions=None,
    # model cards list msmarco; scores distilled from mixedbread-ai/mxbai-rerank-large-v2
    training_datasets={"MSMARCO"},
    public_training_code="https://github.com/xpmir/mice",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    citation=mice_p_citation,
)

mice_p_61m = ModelMeta(
    **_MICE_P_COMMON,
    name="xpmir/Mice-P-61m",
    revision="98537b2aae05f2208c216adf96e0c8e63776e299",
    n_parameters=61_328_385,
    n_embedding_parameters=25_788_416,
    memory_usage_mb=234,
    adapted_from="jhu-clsp/ettin-encoder-68m",
    reference="https://huggingface.co/xpmir/Mice-P-61m",
)

mice_p_130m = ModelMeta(
    **_MICE_P_COMMON,
    name="xpmir/Mice-P-130m",
    revision="206e826d1225f0f440942e9ff8b973ebe4d9c135",
    n_parameters=128_564_737,
    n_embedding_parameters=38_682_624,
    memory_usage_mb=490,
    adapted_from="cross-encoder/ettin-reranker-150m-v1",
    reference="https://huggingface.co/xpmir/Mice-P-130m",
)
