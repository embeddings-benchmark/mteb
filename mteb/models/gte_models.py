from __future__ import annotations

from functools import partial

import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.instruct_wrapper import instruct_wrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"Instruct: {instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else ""
    )


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_weights=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
    n_parameters=7_613_000_000,
    memory_usage_mb=29040,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    max_tokens=32_768,
)

gte_Qwen1_5_7B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="07d27e5226328010336563bc1b564a5e3436a298",
    release_date="2024-04-20",  # initial commit of hf model.
    n_parameters=7_720_000_000,
    memory_usage_mb=29449,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32_768,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

gte_Qwen2_1_5B_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        instruction_template=instruction_template,
        attn="bbcc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
        embed_eos="<|endoftext|>",
    ),
    name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c6c1b92f4a3e1b92b326ad29dd3c8433457df8dd",
    release_date="2024-07-29",  # initial commit of hf model.
    n_parameters=1_780_000_000,
    memory_usage_mb=6776,
    embed_dim=8960,
    license="apache-2.0",
    max_tokens=32_768,
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

gte_small_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="thenlper/gte-small-zh",
        revision="af7bd46fbb00b3a6963c8dd7f1786ddfbfbe973a",
    ),
    name="thenlper/gte-small-zh",
    languages=["zho_Hans"],
    open_weights=True,
    revision="af7bd46fbb00b3a6963c8dd7f1786ddfbfbe973a",
    release_date="2023-11-08",  # initial commit of hf model.
    n_parameters=int(30.3 * 1e6),
    memory_usage_mb=58,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/thenlper/gte-small-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # Not disclosed
)

gte_base_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="thenlper/gte-base-zh",
        revision="71ab7947d6fac5b64aa299e6e40e6c2b2e85976c",
    ),
    name="thenlper/gte-base-zh",
    languages=["zho_Hans"],
    open_weights=True,
    revision="71ab7947d6fac5b64aa299e6e40e6c2b2e85976c",
    release_date="2023-11-08",  # initial commit of hf model.
    n_parameters=int(102 * 1e6),
    memory_usage_mb=195,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/thenlper/gte-base-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # Not disclosed
)

gte_large_zh = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="thenlper/gte-large-zh",
        revision="64c364e579de308104a9b2c170ca009502f4f545",
    ),
    name="thenlper/gte-large-zh",
    languages=["zho_Hans"],
    open_weights=True,
    revision="64c364e579de308104a9b2c170ca009502f4f545",
    release_date="2023-11-08",  # initial commit of hf model.
    n_parameters=int(326 * 1e6),
    memory_usage_mb=621,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/thenlper/gte-large-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # Not disclosed
)

gte_multilingual_langs = [
    "afr_Latn",
    "ara_Arab",
    "aze_Latn",
    "bel_Cyrl",
    "bul_Cyrl",
    "ben_Beng",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "spa_Latn",
    "est_Latn",
    "eus_Latn",
    "fas_Arab",
    "fin_Latn",
    "fra_Latn",
    "glg_Latn",
    "guj_Gujr",
    "heb_Hebr",
    "hin_Deva",
    "hrv_Latn",
    "hat_Latn",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "jav_Latn",
    "kat_Geor",
    "kaz_Cyrl",
    "khm_Khmr",
    "kan_Knda",
    "kor_Hang",
    "kir_Cyrl",
    "lao_Laoo",
    "lit_Latn",
    "lav_Latn",
    "mkd_Cyrl",
    "mal_Mlym",
    "mon_Cyrl",
    "mar_Deva",
    "msa_Latn",
    "mya_Mymr",
    "nep_Deva",
    "nld_Latn",
    "nor_Latn",
    "pan_Guru",
    "pol_Latn",
    "por_Latn",
    "que_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "swa_Latn",
    "tam_Taml",
    "tel_Telu",
    "tha_Thai",
    "tgl_Latn",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "vie_Latn",
    "yor_Latn",
    "zho_Hans",
]
# Source: https://arxiv.org/pdf/2407.19669
gte_multi_training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "NQ-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "MSMARCO": ["train"],
    "mMARCO-NL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQA-NL": ["train"],
    "FEVER": ["train"],
    "FEVER-NL": ["train"],
    "MrTidyRetrieval": ["train"],
    "MultiLongDocRetrieval": ["train"],
    "MIRACLReranking": ["train"],
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": [
        "train"
    ],  # https://arxiv.org/pdf/2407.19669, Table 11
    # not in MTEB:
    #   - TriviaQA
    #   - SQuAD
    #   - AllNLI
    #   - Multi-CPR
}

gte_multilingual_base = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Alibaba-NLP/gte-multilingual-base",
        revision="ca1791e0bcc104f6db161f27de1340241b13c5a4",
    ),
    name="Alibaba-NLP/gte-multilingual-base",
    languages=gte_multilingual_langs,
    open_weights=True,
    revision="ca1791e0bcc104f6db161f27de1340241b13c5a4",
    release_date="2024-07-20",  # initial commit of hf model.
    n_parameters=int(305 * 1e6),
    memory_usage_mb=582,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,  # couldn't find
    training_datasets=gte_multi_training_data,
)

gte_modernbert_base = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="Alibaba-NLP/gte-modernbert-base",
        revision="7ca8b4ca700621b67618669f5378fe5f5820b8e4",
    ),
    name="Alibaba-NLP/gte-modernbert-base",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7ca8b4ca700621b67618669f5378fe5f5820b8e4",
    release_date="2025-01-21",  # initial commit of hf model.
    n_parameters=int(149 * 1e6),
    memory_usage_mb=284,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-modernbert-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,  # couldn't find
    public_training_data=None,
    training_datasets=gte_multi_training_data,  # English part of gte_multi_training_data,
)


gte_base_en_v15 = ModelMeta(
    name="Alibaba-NLP/gte-base-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a829fd0e060bb84554da0dfd354d0de0f7712b7f",  # can be any
    release_date="2024-06-20",  # initial commit of hf model
    n_parameters=137_000_000,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
