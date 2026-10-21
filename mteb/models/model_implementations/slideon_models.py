from mteb.models.model_implementations.pylate_models import (
    mdenseon_mlateon_code_languages,
    mdenseon_mlateon_languages,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SparseEncoderWrapper
from mteb.types import OutputDType

slideon_training_data = {
    "FiQA2018",
    "HotpotQA",
    "NQ",
    "MSMARCO",
    "FEVER",
    "MIRACLRetrieval",
    "MultiLongDocRetrieval",
}

mslideon_training_data = slideon_training_data | {
    "AppsRetrieval",
    "CosQA",
    "SyntheticText2SQL",
    "CodeFeedbackST",
    "CodeFeedbackMT",
    "StackOverflowQA",
    "CodeTransOceanContest",
    "CodeTransOceanDL",
    "CodeSearchNetRetrieval",
    "CodeSearchNetCCRetrieval",
    "COIRCodeSearchNetRetrieval",
    "CodeEditSearchRetrieval",
}

lightonai_slideon = ModelMeta(
    loader=SparseEncoderWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {"dtype": OutputDType.BF16},
    },
    name="lightonai/SlideOn",
    revision="370c600e9886afc43ff8026695261e56ecfaa011",
    model_type=["sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    release_date=None,
    n_parameters=174_246_400,
    n_embedding_parameters=38_682_624,
    memory_usage_mb=381,
    max_tokens=8192,
    embed_dim=32_768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/SlideOn",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    output_dtypes=[OutputDType.BF16],
    use_instructions=False,
    adapted_from="lightonai/DenseOn",
    public_training_code="https://github.com/lightonai/slideon-mslideon",
    public_training_data="https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-en",
    training_datasets=slideon_training_data,
)

lightonai_mslideon = ModelMeta(
    loader=SparseEncoderWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {"dtype": OutputDType.BF16},
    },
    name="lightonai/mSlideOn",
    revision="ef8016025bfd0ea10e44ee27c4038a5852b1018a",
    model_type=["sparse"],
    languages=mdenseon_mlateon_languages + mdenseon_mlateon_code_languages,
    open_weights=True,
    release_date=None,
    n_parameters=339_449_472,
    n_embedding_parameters=196_609_536,
    memory_usage_mb=701,
    max_tokens=8192,
    embed_dim=32_768,
    license="apache-2.0",
    reference="https://huggingface.co/lightonai/mSlideOn",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    output_dtypes=[OutputDType.BF16],
    use_instructions=False,
    adapted_from="lightonai/mLateOn",
    public_training_code="https://github.com/lightonai/slideon-mslideon",
    public_training_data="https://huggingface.co/collections/lightonai/slideon-and-mslideon",
    training_datasets=mslideon_training_data,
)
