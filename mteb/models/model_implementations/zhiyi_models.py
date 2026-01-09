"""
    Author:     code by zhiyi
    Date:       2026-01-04 18:41:12
    Env:        Python 3.10
    Function:   
    Reference:  
        
"""
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol, PromptType
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel

from mteb.models.model_implementations.kalm_models import KaLM_Embedding_gemma_3_12b_training_data


def instruct_template(instruction: str, prompt_type: PromptType | None = None) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


zhiyi_training_data = {
    "simclue_public",
    "nli_for_simcse",
    "nli_zh_all",
    "MATINF",
    "wikipedia",
    "wudao",
    "webtextqa_zh"
    "T2Retrieval",
    "DuRetrieval",
    "MSMARCO",
    "HotpotQA",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "CodeSearchNet",
    "Zhihu-KOL"
}
zhiyi_training_data += KaLM_Embedding_gemma_3_12b_training_data


def zhiyi_embedding_loader(model_name_or_path: str, revision: str, **kwargs) -> EncoderProtocol:
    model = InstructSentenceTransformerModel(
        model_name=model_name_or_path,
        revision=revision,
        instruction_template=instruct_template,
        apply_instruction_to_passages=False,
        **kwargs
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model


zhiyi_Embedding_8B = ModelMeta(
    loader=zhiyi_embedding_loader,
    name="Zhiyi-AI/zhiyi-Embedding-8B",
    model_type=["dense"],
    languages=["zho-Hans"],
    open_weights=False,
    revision='8d7f3a2c9e1b5f0a4c8e2d6b9f3a5c7e1b9d5f2a',
    release_date="2026-01-04",
    n_parameters=7_567_295_488,
    memory_usage_mb=14563,
    max_tokens=8192,
    embed_dim=4096,
    license=None,
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=zhiyi_training_data,
    public_training_code=None,
    public_training_data=None,
)
