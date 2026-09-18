from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType, PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"Instruct: {instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else ""
    )


sante_embed = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        add_eos_token=True,
        trust_remote_code=True,
        # the checkpoint truncates at 512; leaving this unset lets the wrapper read
        # max_position_embeddings (131072) and the long-document tasks then OOM
        max_seq_length=512,
        model_kwargs={"dtype": OutputDType.FLOAT16},
        # the remote modeling_qwen.py calls DynamicCache.get_usable_length(),
        # removed in transformers>=4.56; encoding never needs the KV cache
        config_kwargs={"use_cache": False},
    ),
    name="Singaraj/sante-embed",
    model_type=["dense"],
    languages=[
        "eng-Latn",
        "zho-Hans",
        "ara-Arab",
        "spa-Latn",
        "fra-Latn",
        "kor-Hang",
        "pol-Latn",
        "rus-Cyrl",
        "vie-Latn",
    ],
    open_weights=True,
    revision="9343a6f818846268da4a47afb6c8550bdc608a7c",
    release_date="2026-09-15",
    n_parameters=1_543_268_864,
    n_embedding_parameters=232_928_256,
    memory_usage_mb=2944,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/Singaraj/sante-embed",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    superseded_by=None,
    adapted_from="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    training_datasets={"NFCorpus", "SciFact"},
    public_training_code=None,
    public_training_data=None,
)
