"""
Models from ICT-TIME-and-Querit organization.

这个文件应该位置在：mteb/models/model_implementations/ict_time.py
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
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))  # TODO
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"

multilingual_langs = ["deu-Latn","ita-Latn", "ara-Arab", "fas-Arab", "fra-Latn", "hin-Deva", "spa-Latn", "zho-Hans",  "ben-Beng", "eng-Latn", "fin-Latn",  "ind-Latn", "jpn-Jpan", "kor-Hang", "rus-Cyrl", "swh", "tel-Telu", "tha-Thai"], 

training_data = {"FEVER", "CornStack",  "DuRetrieval", "HotpotQA", "MSMARCO", "T2Retrieval", "NQ", "MIRACLRetrieval", "MrTidyRetrieval", "amazon_counterfactual", "amazon_reviews", "banking", "emotion", "IMDB", "Mtop_intent", "toxic_conversations", "tweet_sentiment", "eli5", "nli", "squad", "trivial"}


def q3e_instruct_loader(
    model_name_or_path: str, revision: str, **kwargs
) -> EncoderProtocol:
    model = InstructSentenceTransformerModel(
        model_name_or_path,
        revision=revision,
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model

boom_4b_v1 = ModelMeta(
    loader=q3e_instruct_loader,
    name="ICT-TIME-and-Querit/BOOM_4B_v1",
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    revision="33fb345468120e37c81eed2369aefe08b8f8222b", 
    release_date="2026-01-31", 
    n_parameters=4021774336,
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

