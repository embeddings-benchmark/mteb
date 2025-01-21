from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import instruct_wrapper

stella_en_400M = ModelMeta(
    # https://huggingface.co/dunzhang/stella_en_400M_v5/discussions/21#671a6205ac1e2416090f2bf4
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="dunzhang/stella_en_400M_v5",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="dunzhang/stella_en_400M_v5",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    revision="1bb50bc7bb726810eac2140e62155b88b0df198f",
    release_date="2024-07-12",
    n_parameters=435_000_000,
    max_tokens=8192,
    embed_dim=4096,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "GritLM"],
    reference="https://huggingface.co/dunzhang/stella_en_400M_v5",
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

stella_en_1_5b = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="dunzhang/stella_en_1.5B_v5",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="dunzhang/stella_en_1.5B_v5",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    revision="d03be74b361d4eb24f42a2fe5bd2e29917df4604",
    release_date="2024-07-12",
    n_parameters=1_540_000_000,
    max_tokens=131072,
    embed_dim=8960,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "GritLM"],
    reference="https://huggingface.co/dunzhang/stella_en_1.5B_v5",
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

stella_large_zh_v3_1792d = ModelMeta(
    name="dunzhang/stella-large-zh-v3-1792d",
    languages=["zho_Hans"],
    open_weights=True,
    revision="d5d39eb8cd11c80a63df53314e59997074469f09",
    release_date="2024-02-17",
    n_parameters=None,  # can't see on model card
    embed_dim=1792,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/dunzhang/stella-large-zh-v3-1792d",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="dunzhang/stella-mrl-large-zh-v3.5-1792d",
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # Not in MTEB:
        # - infgrad/dialogue_rewrite_llm
        # - infgrad/retrieval_data_llm
    },
)

stella_base_zh_v3_1792d = ModelMeta(
    name="infgrad/stella-base-zh-v3-1792d",
    languages=["zho_Hans"],
    open_weights=True,
    revision="82254892a0fba125aa2abf3a4800d2dd12821343",
    release_date="2024-02-17",
    n_parameters=None,  # can't see on model card
    embed_dim=1792,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/infgrad/stella-base-zh-v3-1792d",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # Not in MTEB:
        # - infgrad/dialogue_rewrite_llm
        # - infgrad/retrieval_data_llm
    },
)


stella_mrl_large_zh_v3_5_1792d = ModelMeta(
    name="dunzhang/stella-mrl-large-zh-v3.5-1792d",
    languages=["zho_Hans"],
    open_weights=True,
    revision="17bb1c32a93a8fc5f6fc9e91d5ea86da99983cfe",
    release_date="2024-02-27",
    n_parameters=326 * 1e6,
    embed_dim=1792,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/dunzhang/stella-large-zh-v3-1792d",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="dunzhang/stella-large-zh-v3-1792d",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # Not specified
)

zpoint_large_embedding_zh = ModelMeta(
    name="iampanda/zpoint_large_embedding_zh",
    languages=["zho_Hans"],
    open_weights=True,
    revision="b1075144f440ab4409c05622c1179130ebd57d03",
    release_date="2024-06-04",
    n_parameters=326 * 1e6,
    embed_dim=1792,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/iampanda/zpoint_large_embedding_zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="dunzhang/stella-mrl-large-zh-v3.5-1792d",
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # It's a bit unclear what they have trained on to be honest, because they don't list all
        # And they also have some rather cryptic description of their training procedure, but at
        # Least they disclose that they have trained on these:
        "MIRACLRetrieval": ["train"],
        "MIRACLReranking": ["train"],
        "DuRetrieval": ["train"],
        "T2Retrieval": ["train"],
        "MultiLongDocRetrieval": ["train"],
        #  Not in MTEB:
        #  - Shitao/bge-reranker-data
        #  - FreedomIntelligence/Huatuo26M-Lite
    },
)
