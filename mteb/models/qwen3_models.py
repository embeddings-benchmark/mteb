from __future__ import annotations

import os
from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import instruct_wrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.passage:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]  # TODO
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "NQ": ["train"],
    "MSMARCO": ["train"],
    "HotpotQA": ["train"],
    "FEVER": ["train"],
    "MrTidyRetrieval": ["train"],
    "MIRACLRetrieval": ["train"],
    "CodeSearchNet": ["train"],
}


def q3e_instruct_loader(model_name_or_path, **kwargs):
    model = instruct_wrapper(
        model_name_or_path,
        mode="embedding",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        torch_dtype=torch.float16,
        normalized=True,
        embed_eos="<|endoftext|>",
        **kwargs,
    )
    if model.model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        model.tokenizer.padding_side = "left"
    return model


Qwen3_Embedding_0B6 = ModelMeta(
    loader=partial(  # type: ignore
        q3e_instruct_loader,
        model_name_or_path=os.environ.get("Q3E_0B6_PATH", "Qwen/Qwen3-Embedding-0.6B"),
    ),
    name="Qwen/Qwen3-Embedding-0.6B",
    languages=multilingual_langs,
    open_weights=True,
    revision="TODO",
    release_date="2025-06-04",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)

Qwen3_Embedding_4B = ModelMeta(
    loader=partial(  # type: ignore
        q3e_instruct_loader,
        model_name_or_path=os.environ.get("Q3E_4B_PATH", "Qwen/Qwen3-Embedding-4B"),
    ),
    name="Qwen/Qwen3-Embedding-4B",
    languages=multilingual_langs,
    open_weights=True,
    revision="TODO",
    release_date="2025-06-04",
    n_parameters=4021774336,
    memory_usage_mb=15341,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)

Qwen3_Embedding_8B = ModelMeta(
    loader=partial(  # type: ignore
        q3e_instruct_loader,
        model_name_or_path=os.environ.get("Q3E_8B_PATH", "Qwen/Qwen3-Embedding-8B"),
    ),
    name="Qwen/Qwen3-Embedding-8B",
    languages=multilingual_langs,
    open_weights=True,
    revision="TODO",
    release_date="2025-06-04",
    n_parameters=7567295488,
    memory_usage_mb=28866,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)


def test_model():
    queries = [
        "黑龙江的省会在哪儿",
        "Where is the caption of Heilongjiang",
        "阿里巴巴总部在杭州吗",
    ]
    documents = ["阿里巴巴", "黑龙江的省会是哈尔滨", " You are a hero"]
    model = Qwen3_Embedding_0B6.load_model()
    vd = model.encode(documents, task_name="MSMARCO", prompt_type=PromptType.passage)
    vq = model.encode(queries, task_name="MSMARCO", prompt_type=PromptType.query)
    print("query outputs", vq)
    print("doc outputs", vd)
    scores = (vq @ vd.T) * 100
    print(scores.tolist())
    return


if __name__ == "__main__":
    test_model()
