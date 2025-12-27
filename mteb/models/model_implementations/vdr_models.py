from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return "{instruction}"


vdr_languages = [
    "eng-Latn",
    "ita-Latn",
    "fra-Latn",
    "deu-Latn",
    "spa-Latn",
]

vdr_2b_multi_v1 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        max_seq_length=32768,
        apply_instruction_to_passages=True,
    ),
    name="llamaindex/vdr-2b-multi-v1",
    model_type=["dense"],
    languages=vdr_languages,
    open_weights=True,
    revision="2c4e54c8db4071cc61fc3c62f4490124e40c37db",
    release_date="2024-01-08",
    modalities=["text"],  # TODO: integrate with image
    n_parameters=2_000_000_000,
    memory_usage_mb=4213,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/llamaindex/vdr-2b-multi-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Sentence Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/llamaindex/vdr-multilingual-train",
    training_datasets=set(
        # llamaindex/vdr-multilingual-train
        "VDRMultilingualRetrieval",
    ),
)
