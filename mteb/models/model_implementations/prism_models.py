from __future__ import annotations

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

# Prism rerankers are LLM-based cross-encoders. The system prompt and instruction
# are baked into the model's chat template and are not user-configurable, so they
# can be used directly through sentence-transformers' CrossEncoder. The default
# activation is sigmoid, yielding relevance scores in (0, 1).
prism_reranker_citation = """
@misc{zhang2026prismreranker,
  title={Prism-Reranker: Beyond Relevance Scoring -- Jointly Producing Contributions and Evidence for Agentic Retrieval},
  author={Dun Zhang},
  year={2026},
  eprint={2604.23734},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2604.23734}
}
"""

_PRISM_COMMON = dict(
    loader=CrossEncoderWrapper,
    loader_kwargs={"trust_remote_code": True},
    model_type=["cross-encoder"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    release_date="2026-04-26",
    # The model card caps usable context at 10K tokens ("Inputs longer than 10K
    # tokens may degrade") despite the larger Qwen3.5 position-embedding window.
    max_tokens=10000,
    license="mit",
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets={
        "MSMARCO",
        "T2Reranking",
        "MIRACLReranking",
    },
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    citation=prism_reranker_citation,
)

prism_qwen35_reranker_0_8b = ModelMeta(
    **_PRISM_COMMON,
    name="infgrad/Prism-Qwen3.5-Reranker-0.8B",
    revision="c60729783cb2f0662c0b054a02964402a211e3f4",
    n_parameters=752_393_024,
    n_embedding_parameters=254_279_680,
    embed_dim=1024,
    memory_usage_mb=1435,
    reference="https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-0.8B",
)

prism_qwen35_reranker_2b = ModelMeta(
    **_PRISM_COMMON,
    name="infgrad/Prism-Qwen3.5-Reranker-2B",
    revision="29a527157a31fbcc38c8fa10d3628b5cce203c5c",
    n_parameters=1_881_825_088,
    n_embedding_parameters=508_559_360,
    embed_dim=2048,
    memory_usage_mb=3589,
    reference="https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-2B",
)

prism_qwen35_reranker_4b = ModelMeta(
    **_PRISM_COMMON,
    name="infgrad/Prism-Qwen3.5-Reranker-4B",
    revision="bfa8c64ffd688abd4a4dc2c724dd1d83a60a6e92",
    n_parameters=4_205_751_296,
    n_embedding_parameters=635_699_200,
    embed_dim=2560,
    memory_usage_mb=8022,
    reference="https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-4B",
)

prism_qwen35_reranker_9b = ModelMeta(
    **_PRISM_COMMON,
    name="infgrad/Prism-Qwen3.5-Reranker-9B",
    revision="1c40be94424cde2e8a3be44487492e63861e1601",
    n_parameters=8_953_803_264,
    n_embedding_parameters=1_017_118_720,
    embed_dim=4096,
    memory_usage_mb=17078,
    reference="https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-9B",
)
