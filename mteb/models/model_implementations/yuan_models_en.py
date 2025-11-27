from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol, PromptType


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


multilingual_langs = [
      "eng-Latn"
]


training_data = {"T2Retrieval","DuRetrieval","MMarcoReranking","CMedQAv2-reranking","NQ","MSMARCO","HotpotQA","MrTidyRetrieval","MIRACLRetrieval","CodeSearchNet",
}


def yuan_instruct_loader(
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


yuan_embedding_2_en  = ModelMeta(
    loader=yuan_instruct_loader,
    name="IEITYuan/Yuan-embedding-2.0-en",
    languages=multilingual_langs,
    open_weights=True,
    revision="b2fd15da3bcae3473c8529593825c15068f09fce",  
    release_date="2025-11-27",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=1024,
    max_tokens=2048,
    license="apache-2.0",
    reference="https://huggingface.co/IEITYuan/Yuan-embedding-2.0-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
