"""denserepr code-retrieval embedding models."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

# Trained on the train splits of the ten CoIR datasets (decontaminated against their test splits).
# CodeSearchNetRetrieval shares its source data with COIRCodeSearchNetRetrieval.
_COIR_TRAIN = {
    "AppsRetrieval",
    "CodeFeedbackMT",
    "CodeFeedbackST",
    "CodeSearchNetCCRetrieval",
    "CodeSearchNetRetrieval",
    "COIRCodeSearchNetRetrieval",
    "CodeTransOceanContest",
    "CodeTransOceanDL",
    "CosQA",
    "StackOverflowQA",
    "SyntheticText2SQL",
}

denserepr_code_48m_pretrain_en_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    # fp32 weights, evaluated in bf16 (mixed-precision training: fp32 weights, bf16 autocast)
    loader_kwargs={"model_kwargs": {"dtype": "bfloat16"}},
    name="f4roukb/denserepr-code-48M-pretrain-EN-v1",
    model_type=["dense"],
    languages=[
        "eng-Latn",
        "python-Code",
        "go-Code",
        "java-Code",
        "javascript-Code",
        "ruby-Code",
        "php-Code",
        "sql-Code",
    ],
    open_weights=True,
    revision="7e51c172e696db69beee810a36df8ca3799606a5",
    release_date="2026-11-14",
    n_parameters=47_662_464,
    n_embedding_parameters=19_341_312,
    memory_usage_mb=182,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/f4roukb/denserepr-code-48M-pretrain-EN-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=_COIR_TRAIN,
    adapted_from="ibm-granite/granite-embedding-small-english-r2",
    contacts=["f4roukb"],
)
