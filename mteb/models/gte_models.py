from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import instruct_wrapper


def instruction_template(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_weights=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
    n_parameters=7_613_000_000,
    memory_usage=None,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Qwen/Qwen2-7B",
    supersedes="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
)


gte_Qwen15_7B_instruct = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        instruction_template="Instruct: {instruction}\nQuery: ",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    languages=None,
    open_weights=True,
    revision="07d27e5226328010336563bc1b564a5e3436a298",
    release_date="2024-06-20",  # initial commit of hf model
    n_parameters=7_613_000_000,
    memory_usage=None,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Qwen/Qwen1.5-7B",
    supersedes=None,
)

gte_base_en_v15 = ModelMeta(
    name="Alibaba-NLP/gte-base-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a829fd0e060bb84554da0dfd354d0de0f7712b7f",  # can be any
    release_date="2024-06-20",  # initial commit of hf model
    n_parameters=137_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    supersedes=None,
    adapted_from=None,
)

gte_multilingual_base = ModelMeta(
    name="Alibaba-NLP/gte-multilingual-base",
    # lang codes from model card and converted to three letter codes
    languages=[
        "afr-Latn",
        "ara-Latn",
        "aze-Latn",
        "bel-Latn",
        "bul-Latn",
        "ben-Latn",
        "cat-Latn",
        "ceb-Latn",
        "ces-Latn",
        "cym-Latn",
        "dan-Latn",
        "deu-Latn",
        "ell-Latn",
        "eng-Latn",
        "spa-Latn",
        "est-Latn",
        "eus-Latn",
        "fas-Latn",
        "fin-Latn",
        "fra-Latn",
        "glg-Latn",
        "guj-Latn",
        "heb-Latn",
        "hin-Latn",
        "hrv-Latn",
        "hat-Latn",
        "hun-Latn",
        "hye-Latn",
        "ind-Latn",
        "isl-Latn",
        "ita-Latn",
        "jpn-Latn",
        "jav-Latn",
        "kat-Latn",
        "kaz-Latn",
        "khm-Latn",
        "kan-Latn",
        "kor-Latn",
        "kir-Latn",
        "lao-Latn",
        "lit-Latn",
        "lav-Latn",
        "mkd-Latn",
        "mal-Latn",
        "mon-Latn",
        "mar-Latn",
        "msa-Latn",
        "mya-Latn",
        "nep-Latn",
        "nld-Latn",
        "nor-Latn",
        "pan-Latn",
        "pol-Latn",
        "por-Latn",
        "que-Latn",
        "ron-Latn",
        "rus-Latn",
        "sin-Latn",
        "slk-Latn",
        "slv-Latn",
        "som-Latn",
        "sqi-Latn",
        "srp-Latn",
        "swe-Latn",
        "swa-Latn",
        "tam-Latn",
        "tel-Latn",
        "tha-Latn",
        "tgl-Latn",
        "tur-Latn",
        "ukr-Latn",
        "urd-Latn",
        "vie-Latn",
        "yor-Latn",
        "zho-Latn",
    ],
    open_weights=True,
    revision="a829fd0e060bb84554da0dfd354d0de0f7712b7f",  # can be any
    release_date="2024-06-20",  # initial commit of hf model
    n_parameters=137_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    supersedes=None,
    adapted_from=None,
)


gte_Qwen1_5_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="07d27e5226328010336563bc1b564a5e3436a298",
    release_date="2024-04-20",  # initial commit of hf model.
    n_parameters=7_720_000_000,
    memory_usage=None,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)


gte_Qwen2_1_5B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        instruction_template=instruction_template,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c6c1b92f4a3e1b92b326ad29dd3c8433457df8dd",
    release_date="2024-07-29",  # initial commit of hf model.
    n_parameters=1_780_000_000,
    memory_usage=None,
    embed_dim=8960,
    license="apache-2.0",
    max_tokens=131072,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)
