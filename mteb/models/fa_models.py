"""Farsi/Persian models for evaluation on the Persian part of MTEB"""

from __future__ import annotations

from mteb.model_meta import ModelMeta

parsbert = ModelMeta(
    name="HooshvareLab/bert-base-parsbert-uncased",
    languages=["fas-Arab"],
    open_weights=True,
    revision="d73a0e2c7492c33bd5819bcdb23eba207404dd19",
    release_date="2021-05-19",
    n_parameters=162_841_344,
    memory_usage_mb=621,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # It's just a base model
        # https://github.com/miras-tech/MirasText/tree/master/MirasText
        # Persian Wikipedia
        # Other data crawled from websites like bigbangpage.com, chetor.com, eligasht.com/blog, digikala.com/mag, and ted.com/talks.
    },
)

bert_zwnj = ModelMeta(
    name="m3hrdadfi/bert-zwnj-wnli-mean-tokens",
    languages=["fas-Arab"],
    open_weights=True,
    revision="b9506ddc579ac8c398ae6dae680401ae0a1a5b23",
    release_date="2021-06-28",
    n_parameters=118_297_344,
    memory_usage_mb=451,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/m3hrdadfi/bert-zwnj-wnli-mean-tokens",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # This model is finetuned from HooshvareLab/bert-base-parsbert-uncased
        "FarsTail": [],
        # https://github.com/m3hrdadfi/sentence-transformers?tab=readme-ov-file
    },
)

roberta_zwnj = ModelMeta(
    name="m3hrdadfi/roberta-zwnj-wnli-mean-tokens",
    languages=["fas-Arab"],
    open_weights=True,
    revision="36f912ac44e22250aee16ea533a4ff8cd848c1a1",
    release_date="2021-06-28",
    n_parameters=118_298_112,
    memory_usage_mb=451,
    embed_dim=768,
    license="not specified",
    max_tokens=514,
    reference="https://huggingface.co/m3hrdadfi/roberta-zwnj-wnli-mean-tokens",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail": [],
        # https://github.com/m3hrdadfi/sentence-transformers?tab=readme-ov-file
    },
)

sentence_transformer_parsbert = ModelMeta(
    name="myrkur/sentence-transformer-parsbert-fa",
    languages=["fas-Arab"],
    open_weights=True,
    revision="72bd0a3557622f0ae08a092f4643609e0b950cdd",
    release_date="2024-12-10",
    n_parameters=162_841_344,
    memory_usage_mb=621,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/myrkur/sentence-transformer-parsbert-fa",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # This model is finetuned from HooshvareLab/bert-base-parsbert-uncased
        # https://huggingface.co/datasets/Gholamreza/pquad
    },
)

tooka_bert_base = ModelMeta(
    name="PartAI/TookaBERT-Base",
    languages=["fas-Arab"],
    open_weights=True,
    revision="fa5ca89df5670700d9325b8872ac65c17cb24582",
    release_date="2024-12-08",
    n_parameters=122_905_344,
    memory_usage_mb=469,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/PartAI/TookaBERT-Base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # It's just a base model
        # https://huggingface.co/datasets/sbunlp/hmblogs-v3
        # https://huggingface.co/datasets/Targoman/TLPC
        # https://huggingface.co/datasets/allenai/MADLAD-400 (cleaned Persian subset)
    },
)

tooka_sbert = ModelMeta(
    name="PartAI/Tooka-SBERT",
    languages=["fas-Arab"],
    open_weights=True,
    revision="5d07f0c543aca654373b931ae07cd197769110fd",
    release_date="2024-12-07",
    n_parameters=353_039_360,
    memory_usage_mb=1347,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/PartAI/Tooka-SBERT",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # This model is finetuned from PartAI/TookaBERT-Large
    },
)

fa_bert = ModelMeta(
    name="sbunlp/fabert",
    languages=["fas-Arab"],
    open_weights=True,
    revision="a0e3973064c97768e121b9b95f21adc94e0ca3fb",
    release_date="2024-10-07",
    n_parameters=124_441_344,
    memory_usage_mb=475,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/sbunlp/fabert",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # It's just a base model
        # https://huggingface.co/datasets/sbunlp/hmblogs-v3
    },
)
