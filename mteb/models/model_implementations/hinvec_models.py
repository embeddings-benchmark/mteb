import logging

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

logger = logging.getLogger(__name__)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


hinvec_training_datasets = {
    "MintakaRetrieval",
    "HindiDiscourseClassification",
    "SentimentAnalysisHindi",
    "MassiveScenarioClassification",
    "MTOPIntentClassification",
    "LinceMTBitextMining",
    "PhincBitextMining",
    "XNLI",
    "MLQARetrieval",
    "FloresBitextMining",
    "AmazonReviewsClassification",
}

Hinvec_bidir = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        trust_remote_code=True,
        max_seq_length=2048,
        padding_side="left",
        add_eos_token=True,
    ),
    name="Sailesh97/Hinvec",
    model_type=["dense"],
    languages=["eng-Latn", "hin-Deva"],
    open_weights=True,
    revision="d4fc678720cc1b8c5d18599ce2d9a4d6090c8b6b",
    release_date="2025-06-19",
    n_parameters=939_591_680,
    memory_usage_mb=3715,
    embed_dim=2048,
    license="cc-by-nc-4.0",
    max_tokens=2048,
    reference="https://huggingface.co/Sailesh97/Hinvec",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=hinvec_training_datasets,
    public_training_code=None,
    public_training_data=None,
)
