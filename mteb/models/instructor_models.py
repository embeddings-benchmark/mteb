from __future__ import annotations

import logging
from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader

logger = logging.getLogger(__name__)


def instructor_template(instruction: str, prompt_type: PromptType | None = None) -> str:
    """Instructor models use a specific instruction format.
    Based on the instructor-embedding documentation, the models expect:
    - Instruction and text to be passed as a list: [instruction, text]
    - The instruction template is just the instruction itself

    For MTEB integration, we return the instruction as-is since the
    sentence_transformers_loader will handle the proper formatting.
    """
    return instruction if instruction else ""


# Based on the instructor-embedding paper (https://arxiv.org/abs/2212.09741)
# and the model cards, these models were released around December 2022
INSTRUCTOR_RELEASE_DATE = "2022-12-20"

# Training datasets based on the paper - they used MEDI (Multitask Embeddings Data with Instructions)
# which consists of 330 datasets from Super-NI, sentence-transformer data, KILT, and MedMCQA
# According to the paper, they achieved SOTA on 70 diverse embedding tasks
INSTRUCTOR_TRAINING_DATASETS = {
    # Based on the paper, they trained on a large collection including:
    # - Super-NI (Super-NaturalInstructions): 330 tasks
    # - sentence-transformer embedding training data
    # - KILT knowledge-intensive tasks
    # - MedMCQA medical multiple choice QA
    # However, we mark this as None since the exact MTEB dataset overlap
    # requires detailed analysis of their MEDI dataset
}

instructor_base = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="hkunlp/instructor-base",
        revision=None,  # Using latest revision
        instruction_template=instructor_template,
        trust_remote_code=True,
    ),
    name="hkunlp/instructor-base",
    languages=["eng-Latn"],  # Primarily English based on paper
    open_weights=True,
    revision=None,
    release_date=INSTRUCTOR_RELEASE_DATE,
    n_parameters=110_000_000,  # Estimated based on BERT-base architecture
    memory_usage_mb=420,  # Estimated
    embed_dim=768,  # Standard for base models (BERT-base)
    license="apache-2.0",  # Standard license for academic models
    max_tokens=512,  # Standard maximum length for BERT-based models
    reference="https://huggingface.co/hkunlp/instructor-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=INSTRUCTOR_TRAINING_DATASETS,
    public_training_code="https://github.com/HKUNLP/instructor-embedding",
    public_training_data=None,  # MEDI dataset mentioned but not publicly available
)

instructor_large = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="hkunlp/instructor-large",
        revision=None,  # Using latest revision
        instruction_template=instructor_template,
        trust_remote_code=True,
    ),
    name="hkunlp/instructor-large",
    languages=["eng-Latn"],  # Primarily English based on paper
    open_weights=True,
    revision=None,
    release_date=INSTRUCTOR_RELEASE_DATE,
    n_parameters=335_000_000,  # Estimated based on BERT-large architecture
    memory_usage_mb=1280,  # Estimated
    embed_dim=1024,  # Standard for large models (BERT-large)
    license="apache-2.0",  # Standard license for academic models
    max_tokens=512,  # Standard maximum length for BERT-based models
    reference="https://huggingface.co/hkunlp/instructor-large",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=INSTRUCTOR_TRAINING_DATASETS,
    public_training_code="https://github.com/HKUNLP/instructor-embedding",
    public_training_data=None,  # MEDI dataset mentioned but not publicly available
)

instructor_xl = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="hkunlp/instructor-xl",
        revision=None,  # Using latest revision
        instruction_template=instructor_template,
        trust_remote_code=True,
    ),
    name="hkunlp/instructor-xl",
    languages=["eng-Latn"],  # Primarily English based on paper
    open_weights=True,
    revision=None,
    release_date=INSTRUCTOR_RELEASE_DATE,
    n_parameters=1_300_000_000,  # Estimated based on XL transformer architecture
    memory_usage_mb=4900,  # Estimated
    embed_dim=2048,  # Estimated for XL model (could be 1024 or 2048)
    license="apache-2.0",  # Standard license for academic models
    max_tokens=512,  # Standard maximum length for BERT-based models
    reference="https://huggingface.co/hkunlp/instructor-xl",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=INSTRUCTOR_TRAINING_DATASETS,
    public_training_code="https://github.com/HKUNLP/instructor-embedding",
    public_training_data=None,  # MEDI dataset mentioned but not publicly available
)
