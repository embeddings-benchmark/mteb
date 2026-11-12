"""RiverRider's embedding models."""

from mteb.models.model_implementations.bge_models import bge_small_en_v1_5
from mteb.models.model_implementations.gte_models import gte_modernbert_base
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

# One parent was distilled from gte-modernbert-base on the training partitions of CoIR's ten datasets.
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

motherlode_code_small_en_v0_1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="RiverRider/motherlode-code-small-en-v0.1",
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
    revision="eb296bfdc7842a525011d4ee68724c49e4f5f8ea",
    release_date="2026-09-27",
    n_parameters=33_360_000,
    n_embedding_parameters=11_720_448,
    memory_usage_mb=127,
    embed_dim=384,
    license="https://huggingface.co/RiverRider/motherlode-code-small-en-v0.1/blob/main/LICENSE",
    max_tokens=512,
    reference="https://huggingface.co/RiverRider/motherlode-code-small-en-v0.1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=_COIR_TRAIN
    | set(bge_small_en_v1_5.training_datasets or set())
    | set(gte_modernbert_base.training_datasets or set()),
    adapted_from="BAAI/bge-small-en-v1.5",
)
