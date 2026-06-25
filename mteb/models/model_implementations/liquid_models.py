from __future__ import annotations

from mteb.models.model_implementations.pylate_models import MultiVectorModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

LFM_LANGUAGES = [
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "fra-Latn",
    "ita-Latn",
    "por-Latn",
    "ara-Arab",
    "swe-Latn",
    "nor-Latn",
    "jpn-Jpan",
    "kor-Kore",
]

LFM_LICENSE = "https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M/blob/main/LICENSE"

LFM_CITATION = """@misc{liquid2025lfm25,
  title={LFM2.5-Embedding and LFM2.5-ColBERT},
  author={Liquid AI},
  year={2025},
  url={https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M}
}"""

lfm_embedding_model_prompts = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "document: ",
}

lfm25_embedding_350m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=lfm_embedding_model_prompts,
    ),
    name="LiquidAI/LFM2.5-Embedding-350M",
    model_type=["dense"],
    languages=LFM_LANGUAGES,
    open_weights=True,
    revision="f35ae2c91d687658dbf1f2b449382f0b019b9808",
    release_date="2026-05-05",
    n_parameters=354_483_968,
    n_embedding_parameters=67_108_864,
    memory_usage_mb=1352,
    max_tokens=512,
    embed_dim=1024,
    license=LFM_LICENSE,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M",
    use_instructions=True,
    adapted_from="LiquidAI/LFM2.5-350M-Base",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=LFM_CITATION,
)

lfm25_colbert_350m = ModelMeta(
    loader=MultiVectorModel,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="LiquidAI/LFM2.5-ColBERT-350M",
    model_type=["late-interaction"],
    languages=LFM_LANGUAGES,
    open_weights=True,
    revision="59633c2e31717b3502343ff566bee9fda3261943",
    release_date="2026-05-20",
    n_parameters=353_322_752,
    n_embedding_parameters=65_947_648,
    memory_usage_mb=1348,
    max_tokens=512,
    embed_dim=128,
    license=LFM_LICENSE,
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT", "safetensors"],
    reference="https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M",
    use_instructions=False,
    adapted_from="LiquidAI/LFM2.5-350M-Base",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=LFM_CITATION,
    extra_requirements_groups=["pylate"],
)
