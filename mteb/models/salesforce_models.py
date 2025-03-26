from __future__ import annotations

from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.instruct_wrapper import (
    InstructSentenceTransformerWrapper,
    instruct_wrapper,
)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


SFR_TRAINING_DATA = {  # inherits from e5
    **E5_MISTRAL_TRAINING_DATA,
    # From previously released blogpost which now have been taken down:
    "FiQA2018": ["train"],
    "FiQA2018-NL": ["train"],  # translation not trained on
    "FEVER": ["train"],
    "FEVERHardNegatives": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    # source: https://github.com/embeddings-benchmark/leaderboard/issues/41
    # qoute: In the realm of Semantic Textual Similarity (STS), it is trained on STS12, STS22, and STSBenchmark
    "STS12": ["train"],
    "STS22": ["train"],
    "STSBenchmark": ["train"],
}

SFR_Embedding_2_R = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Salesforce/SFR-Embedding-2_R",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Salesforce/SFR-Embedding-2_R
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_weights=True,
    revision="91762139d94ed4371a9fa31db5551272e0b83818",
    release_date="2024-06-14",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-2_R",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=SFR_TRAINING_DATA,
)

SFR_Embedding_Code_2B_R = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name="Salesforce/SFR-Embedding-Code-2B_R",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-Code-2B_R",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c73d8631a005876ed5abde34db514b1fb6566973",
    release_date="2025-01-17",  # initial commit of hf model.
    n_parameters=2_610_000_000,
    memory_usage_mb=4986,
    embed_dim=2304,
    license="cc-by-nc-4.0",
    max_tokens=8192,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="google/gemma-2-2b-it",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

SFR_Embedding_Mistral = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Salesforce/SFR-Embedding-Mistral",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-Mistral",
    languages=["eng_Latn"],
    open_weights=True,
    revision="938c560d1c236aa563b2dbdf084f28ab28bccb11",
    release_date="2024-01-24",  # initial commit of hf model.
    n_parameters=7_110_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/Salesforce/SFR-Embedding-Mistral",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=SFR_TRAINING_DATA,
)
