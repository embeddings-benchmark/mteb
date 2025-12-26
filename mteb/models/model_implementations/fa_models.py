"""Farsi/Persian models for evaluation on the Persian part of MTEB"""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

parsbert = ModelMeta(
    loader=sentence_transformers_loader,
    name="HooshvareLab/bert-base-parsbert-uncased",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # It's just a base model
        # https://github.com/miras-tech/MirasText/tree/master/MirasText
        # Persian Wikipedia
        # Other data crawled from websites like bigbangpage.com, chetor.com, eligasht.com/blog, digikala.com/mag, and ted.com/talks.
    ),
    citation="""
    @article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
""",
)

bert_zwnj = ModelMeta(
    loader=sentence_transformers_loader,
    name="m3hrdadfi/bert-zwnj-wnli-mean-tokens",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # This model is finetuned from HooshvareLab/bert-base-parsbert-uncased
        "FarsTail",
        # https://github.com/m3hrdadfi/sentence-transformers?tab=readme-ov-file
    },
)

roberta_zwnj = ModelMeta(
    loader=sentence_transformers_loader,
    name="m3hrdadfi/roberta-zwnj-wnli-mean-tokens",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "FarsTail",
        # https://github.com/m3hrdadfi/sentence-transformers?tab=readme-ov-file
    },
)

sentence_transformer_parsbert = ModelMeta(
    loader=sentence_transformers_loader,
    name="myrkur/sentence-transformer-parsbert-fa",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # This model is finetuned from HooshvareLab/bert-base-parsbert-uncased
        # https://huggingface.co/datasets/Gholamreza/pquad
    ),
)

tooka_bert_base = ModelMeta(
    loader=sentence_transformers_loader,
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # It's just a base model
        # https://huggingface.co/datasets/sbunlp/hmblogs-v3
        # https://huggingface.co/datasets/Targoman/TLPC
        # https://huggingface.co/datasets/allenai/MADLAD-400 (cleaned Persian subset)
    ),
)

tooka_sbert = ModelMeta(
    loader=sentence_transformers_loader,
    name="PartAI/Tooka-SBERT",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation="""@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}""",
)

fa_bert = ModelMeta(
    loader=sentence_transformers_loader,
    name="sbunlp/fabert",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(
        # It's just a base model
        # https://huggingface.co/datasets/sbunlp/hmblogs-v3
    ),
    citation="""@inproceedings{masumi-etal-2025-fabert,
    title = "{F}a{BERT}: Pre-training {BERT} on {P}ersian Blogs",
    author = "Masumi, Mostafa  and
      Majd, Seyed Soroush  and
      Shamsfard, Mehrnoush  and
      Beigy, Hamid",
    editor = "Bak, JinYeong  and
      Goot, Rob van der  and
      Jang, Hyeju  and
      Buaphet, Weerayut  and
      Ramponi, Alan  and
      Xu, Wei  and
      Ritter, Alan",
    booktitle = "Proceedings of the Tenth Workshop on Noisy and User-generated Text",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.wnut-1.10/",
    doi = "10.18653/v1/2025.wnut-1.10",
    pages = "85--96",
    ISBN = "979-8-89176-232-9",
}""",
)

tooka_sbert_v2_small = ModelMeta(
    loader=sentence_transformers_loader,
    name="PartAI/Tooka-SBERT-V2-Small",
    model_type=["dense"],
    languages=["fas-Arab"],
    open_weights=True,
    revision="8bbed87e36669387f71437c061430ba56d1b496f",
    release_date="2025-05-01",
    n_parameters=122_905_344,
    memory_usage_mb=496,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/PartAI/Tooka-SBERT-V2-Small",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation="""@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}""",
)

tooka_sbert_v2_large = ModelMeta(
    loader=sentence_transformers_loader,
    name="PartAI/Tooka-SBERT-V2-Large",
    model_type=["dense"],
    languages=["fas-Arab"],
    open_weights=True,
    revision="b59682efa961122cc0e4408296d5852870c82eae",
    release_date="2025-05-01",
    n_parameters=353_039_360,
    memory_usage_mb=1347,
    embed_dim=1024,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/PartAI/Tooka-SBERT-V2-Large",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation="""@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}""",
)
