from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    """Prefix queries with "Instruct: ...\nQuery:" and leave documents bare.

    Matches the template in the checkpoint's config_sentence_transformers.json.
    """
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        instruction = (
            instruction[prompt_type]
            if prompt_type is not None
            else next(iter(instruction.values()))
        )
    return f"Instruct: {instruction}\nQuery:"


# Fetched verbatim from https://arxiv.org/bibtex/2508.07995 (v5); the copy in
# the model card predates the revision and lists an older author set.
DIVER_CITATION = """@misc{sun2026divermultistageapproachreasoningintensive,
      title={DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval},
      author={Duolin Sun and Meixiu Long and Dan Yang and Junjie Wang and Yecheng Luo and Yue Shen and Jian Wang and Hualei Zhou and Chunxiao Guo and Peng Wei and Jiahai Wang and Jinjie Gu},
      year={2026},
      eprint={2508.07995},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.07995},
}"""

# Every DIVER-Retriever card lists reasonir/reasonir-data among its training
# sets, and the hard-query half of that dataset reconstructs its positive
# documents from xlangai/BRIGHT — so the BRIGHT corpora are part of these
# models' training data too. The other listed sets (MATH_qCoT...,
# truehealth/medqa, AQ-MedAI/PRGB-ZH) are not MTEB tasks.
DIVER_TRAINING_DATA = {
    "BrightAopsRetrieval",
    "BrightBiologyRetrieval",
    "BrightEarthScienceRetrieval",
    "BrightEconomicsRetrieval",
    "BrightLeetcodeRetrieval",
    "BrightPonyRetrieval",
    "BrightPsychologyRetrieval",
    "BrightRoboticsRetrieval",
    "BrightStackoverflowRetrieval",
    "BrightSustainableLivingRetrieval",
    "BrightTheoremQAQuestionsRetrieval",
    "BrightTheoremQATheoremsRetrieval",
}

DIVER_RETRIEVER_4B_1020 = ModelMeta(
    # A Qwen3-Embedding-4B fine-tune, so it keeps that interface: last-token
    # pooling, cosine similarity, and an "Instruct: ...\nQuery:" prefix on
    # queries only (verified against its config_sentence_transformers.json).
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="AQ-MedAI/Diver-Retriever-4B-1020",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="3984377d297f4c3bcf5d625ea7b48feea7f58c57",
    release_date="2025-10-20",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=7670,
    embed_dim=2560,
    max_tokens=40960,
    license="apache-2.0",
    reference="https://huggingface.co/AQ-MedAI/Diver-Retriever-4B-1020",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/AQ-MedAI/Diver",
    public_training_data=None,
    training_datasets=DIVER_TRAINING_DATA,
    adapted_from="Qwen/Qwen3-Embedding-4B",
    citation=DIVER_CITATION,
)

DIVER_RETRIEVER_4B = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="AQ-MedAI/Diver-Retriever-4B",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="ff7f8bc8d1827734dcf2bbfab99f065be618069a",
    release_date="2025-08-22",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=7670,
    embed_dim=2560,
    max_tokens=40960,
    license="apache-2.0",
    reference="https://huggingface.co/AQ-MedAI/Diver-Retriever-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/AQ-MedAI/Diver",
    public_training_data=None,
    training_datasets=DIVER_TRAINING_DATA,
    adapted_from="Qwen/Qwen3-Embedding-4B",
    superseded_by="AQ-MedAI/Diver-Retriever-4B-1020",
    citation=DIVER_CITATION,
)

DIVER_RETRIEVER_1B7 = ModelMeta(
    # Unlike the other DIVER retrievers this one is built on the Qwen3-1.7B
    # base LM rather than a Qwen3-Embedding checkpoint, but it exposes the same
    # interface. Note its 1_Pooling/config.json carries the 4B model's
    # word_embedding_dimension (2560); the checkpoint's hidden size is 2048.
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="AQ-MedAI/Diver-Retriever-1.7B",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="6b3242ce3928c447b66525b4f2240b6df375321b",
    release_date="2025-10-13",
    n_parameters=1720574976,
    n_embedding_parameters=311164928,
    memory_usage_mb=3282,
    embed_dim=2048,
    max_tokens=40960,
    license="apache-2.0",
    reference="https://huggingface.co/AQ-MedAI/Diver-Retriever-1.7B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/AQ-MedAI/Diver",
    public_training_data=None,
    training_datasets=DIVER_TRAINING_DATA,
    adapted_from="Qwen/Qwen3-1.7B-Base",
    citation=DIVER_CITATION,
)

DIVER_RETRIEVER_0B6 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="AQ-MedAI/Diver-Retriever-0.6B",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="9ce2a1e8acae4342c453e1a18b71d468c4c81e39",
    release_date="2025-09-05",
    n_parameters=595776512,
    n_embedding_parameters=155309056,
    memory_usage_mb=1136,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/AQ-MedAI/Diver-Retriever-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/AQ-MedAI/Diver",
    public_training_data=None,
    training_datasets=DIVER_TRAINING_DATA,
    adapted_from="Qwen/Qwen3-Embedding-0.6B",
    citation=DIVER_CITATION,
)
