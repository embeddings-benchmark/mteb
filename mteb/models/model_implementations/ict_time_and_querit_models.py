"""
Models from the ICT-TIME and Querit organizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from mteb.models.models_protocols import EncoderProtocol


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    """Format instruction for the model."""
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))
        else:
            instruction = instruction.get(prompt_type, next(iter(instruction.values())))
    return f"Instruct: {instruction}\nQuery:"


# Multilingual language codes
multilingual_langs = [
    "deu-Latn",
    "ita-Latn",
    "ara-Arab",
    "fas-Arab",
    "fra-Latn",
    "hin-Deva",
    "spa-Latn",
    "zho-Hans",
    "ben-Beng",
    "eng-Latn",
    "fin-Latn",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
]

# Training datasets
training_data = [
    "FEVER",
    "DuRetrieval",
    "HotpotQA",
    "MSMARCO",
    "T2Retrieval",
    "NQ",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "ImdbClassification",
    "MTOPDomainClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]


boom_4b_v1 = ModelMeta(
    loader=InstructSentenceTransformerModel,   
    loader_kwargs=dict(  
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="ICT-TIME-and-Querit/BOOM_4B_v1",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="33fb345468120e37c81eed2369aefe08b8f8222b",
    release_date="2026-01-31",
    n_parameters=4_021_774_336,
    n_embedding_parameters=None,
    memory_usage_mb=7671,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/ICT-TIME-and-Querit/BOOM_4B_v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
