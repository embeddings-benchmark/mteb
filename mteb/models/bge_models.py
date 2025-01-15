from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}
model_prompts_zh = {"query": "为这个句子生成表示以用于检索相关文章："}

bge_m_training_data = {
    # source: https://arxiv.org/pdf/2402.03216
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "LeCaRDv2": ["train"],
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "MrTidyRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
    # + synthetic data
}

bge_training_data = {
    # source: https://data.baai.ac.cn/details/BAAI-MTP
    "NQ": ["test"],
    "NQHardNegatives": ["test"],
    "AmazonReviewsClassification": [
        "validation",
        "test",
    ],  # assumed from: amazon_reviews_multi
    "MLQARetrieval": [
        "validation",
        "test",
    ],  # assumed from mlqa	(question, context)
    # not in mteb
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context)
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
    # "wikipedia": [],  # title + section title, passage
    # "reddit": [],  # title, body
    # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

bge_chinese_training_data = {
    # source: https://arxiv.org/pdf/2309.07597
    "T2Retrieval": ["train"],
    "DuReader": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    # not in mteb
    #  - multi-cpr
    #  - NLI-zh
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context)
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
    # "wikipedia": [],  # title + section title, passage
    # "reddit": [],  # title, body
    # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

bge_small_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
    memory_usage=None,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets={
        # source: https://data.baai.ac.cn/details/BAAI-MTP
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "AmazonReviewsClassification": [
            "validation",
            "test",
        ],  # assumed from: amazon_reviews_multi
        "MLQARetrieval": [
            "validation",
            "test",
        ],  # assumed from mlqa	(question, context)
        # not in mteb
        # Dataset	Pairs
        # wudao	(title, passage)
        # cmrc2018	(query, context)
        # dureader	(query, context)
        # simclue	(sentence_a, sentence_b)
        # csl	(title, abstract)
        # amazon_reviews_multi	(title, body)
        # wiki_atomic_edits	(base_sentence, edited_sentence)
        # mlqa	(question, context)
        # xlsum	(title, summary) (title, text)
        # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
        # "wikipedia": [],  # title + section title, passage
        # "reddit": [],  # title, body
        # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
        # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
    },
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
    memory_usage=None,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets={
        # source: https://data.baai.ac.cn/details/BAAI-MTP
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "AmazonReviewsClassification": [
            "validation",
            "test",
        ],  # assumed from: amazon_reviews_multi
        "MLQARetrieval": [
            "validation",
            "test",
        ],  # assumed from mlqa	(question, context)
        # not in mteb
        # Dataset	Pairs
        # wudao	(title, passage)
        # cmrc2018	(query, context)
        # dureader	(query, context)
        # simclue	(sentence_a, sentence_b)
        # csl	(title, abstract)
        # amazon_reviews_multi	(title, body)
        # wiki_atomic_edits	(base_sentence, edited_sentence)
        # mlqa	(question, context)
        # xlsum	(title, summary) (title, text)
        # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
        # "wikipedia": [],  # title + section title, passage
        # "reddit": [],  # title, body
        # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
        # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
    },
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets={
        # source: https://data.baai.ac.cn/details/BAAI-MTP
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "AmazonReviewsClassification": [
            "validation",
            "test",
        ],  # assumed from: amazon_reviews_multi
        "MLQARetrieval": [
            "validation",
            "test",
        ],  # assumed from mlqa	(question, context)
        # not in mteb
        # Dataset	Pairs
        # wudao	(title, passage)
        # cmrc2018	(query, context)
        # dureader	(query, context)
        # simclue	(sentence_a, sentence_b)
        # csl	(title, abstract)
        # amazon_reviews_multi	(title, body)
        # wiki_atomic_edits	(base_sentence, edited_sentence)
        # mlqa	(question, context)
        # xlsum	(title, summary) (title, text)
        # "sentence-transformers data": [],  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
        # "wikipedia": [],  # title + section title, passage
        # "reddit": [],  # title, body
        # "stackexchange": [],  # (title, upvoted answer) (title+body, upvoted answer)
        # "s2orc": [],  # (title, abstract) (title, citation title) (abstract, citation abstract)
    },
)

bge_small_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh-v1.5",
    languages=["zho_Hans"],
    open_weights=True,
    revision="7999e1d3359715c523056ef9478215996d62a620",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
    memory_usage=None,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets=bge_chinese_training_data,
)

bge_base_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-zh-v1.5",
        revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-base-zh-v1.5",
    languages=["zho_Hans"],
    open_weights=True,
    revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
    memory_usage=None,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets=bge_chinese_training_data,
)

bge_large_zh_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-zh-v1.5",
        revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-large-zh-v1.5",
    languages=["zho_Hans"],
    open_weights=True,
    revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-zh-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_data=True,  # https://data.baai.ac.cn/details/BAAI-MTP
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    training_datasets=bge_chinese_training_data,
)

bge_multilingual_gemma2 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-multilingual-gemma2",
        revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    ),
    name="BAAI/bge-multilingual-gemma2",
    languages=[
        "eng_Latn",
        "zho_Hans",
        "kor_Hang",
        "kor_Latn",
        "fra_Latn",
        "jpn_Jpan",
        "jpn_Latn",
    ],  # This list is incomlete. Their description says "and more".
    # I'm also unsure about the scripts.
    open_weights=True,
    revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=9.24 * 1e9,
    memory_usage=None,
    embed_dim=3584,  # from old C-MTEB leaderboard
    license="gemma",
    max_tokens=8192,  # from old C-MTEB leaderboard
    reference="https://huggingface.co/BAAI/bge-multilingual-gemma2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_data=False,
    public_training_code=False,
    training_datasets=None,  # not disclosed
)
