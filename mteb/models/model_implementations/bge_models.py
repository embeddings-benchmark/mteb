from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

from .e5_instruct import E5_MISTRAL_TRAINING_DATA

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}
BGE_15_CITATION = """@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding},
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""
model_prompts_zh = {"query": "为这个句子生成表示以用于检索相关文章："}

bge_m3_training_data = {
    # source: https://arxiv.org/abs/2402.03216
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",
    "LeCaRDv2",
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "MrTidyRetrieval",
    "T2Reranking",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQ-NL",  # translation not trained on
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    "HotpotQAHardNegatives",
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CodeSearchNet",
    # not in mteb
    # "s2orc"
    # Wikipedia
    # "xP3"
    # "mC4"
    # "CC-News"
    # "MTP"
    # "NLLB"
    # "CCMatrix"
    # TriviaQA
    # COL-IEE
    # PubMedQA
    # SQuAD
    # SimCSE
    # mMARCO-ZH
    # LawGPT
    # NLI-zh2, LeCaRDv2,
    # NLI, MultiLongDoc (their synthetic)
    # + synthetic data
}

bge_training_data = {
    # source: https://data.baai.ac.cn/details/BAAI-MTP
    "NQ",
    "NQ-NL",  # translation not trained on
    "NQHardNegatives",
    "AmazonReviewsClassification",  # assumed from: amazon_reviews_multi
    "MLQARetrieval",  # assumed from mlqa	(question, context)
    "DuRetrieval",
    # not in mteb
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # "sentence-transformers data",  # https://huggingface.co/datasets/sentence-transformers/embedding-training-data # TODO check this further
    # "wikipedia",  # title + section title, passage
    # "reddit",  # title, body
    # "stackexchange",  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc",  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

bge_chinese_training_data = {
    # source: https://arxiv.org/pdf/2309.07597
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "Cmnli",
    "Ocnli",
    # from https://github.com/FlagOpen/FlagEmbedding/blob/1.1/FlagEmbedding/baai_general_embedding/README.md
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "NQ",
    "NQHardNegatives",
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQAHardNegatives",
    "QuoraRetrieval",
    "QuoraRetrievalHardNegatives",
    "Quora-PLHardNegatives",
    "QuoraRetrieval-Fa",
    "Quora-PL",
    # "StackExchangeClusteringP2P",
    # "StackExchangeClusteringP2P.v2",
    # "StackExchangeClustering",
    # "StackExchangeClustering.v2",
    # not in mteb
    #  - multi-cpr
    #  - NLI-zh
    # Dataset	Pairs
    # wudao	(title, passage)
    # cmrc2018	(query, context)
    # dureader	(query, context) - DuRetrieval
    # simclue	(sentence_a, sentence_b)
    # csl	(title, abstract)
    # amazon_reviews_multi	(title, body)
    # wiki_atomic_edits	(base_sentence, edited_sentence)
    # mlqa	(question, context)
    # xlsum	(title, summary) (title, text)
    # s2orc
    # "wikipedia",  # title + section title, passage
    # "reddit",  # title, body
    # "stackexchange",  # (title, upvoted answer) (title+body, upvoted answer)
    # "s2orc",  # (title, abstract) (title, citation title) (abstract, citation abstract)
}

# https://huggingface.co/BAAI/bge-m3/discussions/29
bgem3_languages = [
    "afr-Latn",  # af
    "amh-Ethi",  # am
    # an
    # ar
    "azj-Latn",  # arz
    # as
    "ast-Latn",  # ast
    # av
    # az
    "azj-Latn",  # azb
    # bar
    # bcl
    "ben-Beng",  # be
    "bul-Cyrl",  # bg
    # bh
    # bn
    # bo
    "bel-Cyrl",  # bpy
    # br
    # bs
    # bxr
    "cat-Latn",  # ca
    # cbk
    # ce
    "ceb-Latn",  # ceb
    "ckb-Arab",  # ckb
    # co
    # cs
    # cv
    # cy
    "dan-Latn",  # da
    "deu-Latn",  # de
    # diq
    # dsb
    # dty
    # dv
    "ell-Grek",  # el
    # eml
    "eng-Latn",  # en
    # eo
    "est-Latn",  # es
    # et
    # eu
    # fa
    "fin-Latn",  # fi
    "fra-Latn",  # fr
    # fy
    # ga
    # gd
    "glg-Latn",  # gl
    # gn
    # gom
    "guj-Gujr",  # gu
    # gv
    "heb-Hebr",  # he
    "hin-Deva",  # hi
    # hif
    # hr
    # hsb
    # ht
    # hu
    # hy
    # ia
    # id
    # ie
    # ilo
    # io
    # is
    "ita-Latn",  # it
    "jpn-Jpan",  # ja
    # jbo
    # jv
    # ka
    # kk
    # km
    # kn
    "kor-Hang",  # ko
    # krc
    # ku
    # kv
    # kw
    # ky
    # la
    # lb
    # lez
    # li
    # lmo
    # lo
    # lt
    # lv
    # mai
    # mg
    # mhr
    # min
    # mk
    # ml
    # mn
    # mr
    # mrj
    # ms
    # mt
    # mwl
    # my
    # myv
    # mzn
    # nah
    # nap
    # nds
    # ne
    # new
    # nl
    # nn
    # no
    # oc
    # or
    # os
    # pa
    # pam
    # pfl
    # pl
    # pms
    # pnb
    # ps
    # pt
    # qu
    # rm
    # ro
    "rus-Cyrl",  # ru
    # sa
    # sah
    # sc
    # scn
    # sco
    # sd
    # sh
    # si
    # sk
    # sl
    # so
    # sq
    # sr
    # su
    # sv
    # sw
    # ta
    # te
    # tg
    "tha-Thai",  # th
    # tk
    # tl
    # tr
    # tt
    # tyv
    # ug
    "ukr-Cyrl",  # uk
    # ur
    # uz
    # vec
    # vep
    # vi
    # vls
    # vo
    # wa
    # war
    # wuu
    # xal
    # xmf
    # yi
    # yo
    # yue
    "zho-Hans",  # zh
]

bge_small_en_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    citation=BGE_15_CITATION,
)

bge_base_en_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    citation=BGE_15_CITATION,
)

bge_large_en_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    citation=BGE_15_CITATION,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
)

bge_small_zh = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-zh",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-small-zh-v1.5",
)

bge_base_zh = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-base-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="0e5f83d4895db7955e4cb9ed37ab73f7ded339b6",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-zh",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-base-zh-v1.5",
)

bge_large_zh = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-large-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8",
    release_date="2023-08-02",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-zh",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
    superseded_by="BAAI/bge-large-zh-v1.5",
)

bge_small_en = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4778d71a06863076696b03fd2777eb118712cad8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-small-en-v1.5",
)

bge_base_en = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=390,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-base-en-v1.5",
)

bge_large_en = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="abe7d9d814b775ca171121fb03f394dc42974275",
    release_date="2023-08-05",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1242,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # seemingly released (at least for some models, but the link is broken
    public_training_data="https://data.baai.ac.cn/details/BAAI-MTP",
    training_datasets=bge_training_data,
    superseded_by="BAAI/bge-large-en-v1.5",
)


bge_small_zh_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-small-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="7999e1d3359715c523056ef9478215996d62a620",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=33_400_000,
    memory_usage_mb=91,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-zh-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_base_zh_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-base-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=109_000_000,
    memory_usage_mb=416,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-zh-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_large_zh_v1_5 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts_zh,
    ),
    name="BAAI/bge-large-zh-v1.5",
    languages=["zho-Hans"],
    open_weights=True,
    revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=1278,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-zh-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=bge_chinese_training_data,
)

bge_m3 = ModelMeta(
    loader=sentence_transformers_loader,
    name="BAAI/bge-m3",
    languages=bgem3_languages,
    open_weights=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-06-28",
    n_parameters=568_000_000,
    memory_usage_mb=2167,
    embed_dim=1024,
    license="mit",
    max_tokens=8194,
    reference="https://huggingface.co/BAAI/bge-m3",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_m3_training_data,
)

# Contents of cfli/bge-full-data
bge_full_data = {
    # source: https://arxiv.org/pdf/2409.15700
    # Charles Goodhart is turning back and forth
    # in his grave as I'm annotating this
    # |Retrieval|
    # ELI5
    # SQuaD
    # TriviaQA
    # QuoraDuplicateQuestions
    "HotpotQA",
    "HotpotQA-NL",  # translation not trained on
    "FEVER",
    "FEVER-NL",  # translation not trained on
    "MSMARCO",
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQ-NL",  # translation not trained on
    "ArguAna",
    "ArguAna-NL",  # translation not trained on
    "FiQA2018",
    "FiQA2018-NL",  # translation not trained on
    # |Reranking|
    "SciDocsReranking",
    "StackOverflowDupQuestions",
    # |Classification|
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "EmotionClassification",
    "TweetSentimentExtractionClassification",
    "MTOPIntentClassification",
    "ImdbClassification",
    "ToxicConversationsClassification",
    # |Clustering|
    "ArxivClusteringS2S",
    "ArxivClusteringP2P",
    "BiorxivClusteringS2S",
    "BiorxivClusteringP2P",
    "MedrxivClusteringS2S",
    "MedrxivClusteringP2P",
    "BiorxivClusteringS2S.v2",
    "BiorxivClusteringP2P.v2",
    "MedrxivClusteringS2S.v2",
    "MedrxivClusteringP2P.v2",
    "RedditClusteringP2P",
    "RedditClustering",
    "RedditClustering.v2",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClustering.v2",
    # |STS|
    "STS22",
    "STS22.v2",
    "STSBenchmark",
}


bge_multilingual_gemma2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="BAAI/bge-multilingual-gemma2",
    languages=[
        "eng-Latn",
        "zho-Hans",
        "kor-Hang",
        "kor-Latn",
        "fra-Latn",
        "jpn-Jpan",
        "jpn-Latn",
    ],  # This list is incomlete. Their description says "and more".
    # I'm also unsure about the scripts.
    open_weights=True,
    revision="992e13d8984fde2c31ef8a3cb2c038aeec513b8a",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=int(9.24 * 1e9),
    memory_usage_mb=35254,
    embed_dim=3584,  # from old C-MTEB leaderboard
    license="https://ai.google.dev/gemma/terms",
    max_tokens=8192,  # from old C-MTEB leaderboard
    reference="https://huggingface.co/BAAI/bge-multilingual-gemma2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "MIRACLReranking",
        "MrTidyRetrieval",
    }
    | bge_full_data
    | bge_m3_training_data,
)

bge_en_icl = ModelMeta(
    loader=sentence_transformers_loader,
    name="BAAI/bge-en-icl",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="971c7e1445cc86656ca0bd85ed770b8675a40bb5",
    release_date="2024-07-25",  # initial commit of hf model.
    n_parameters=int(7.11 * 1e9),
    memory_usage_mb=27125,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/BAAI/bge-en-icl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/FlagOpen/FlagEmbedding",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=E5_MISTRAL_TRAINING_DATA | bge_full_data,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    citation="""
    @misc{li2024makingtextembeddersfewshot,
      title={Making Text Embedders Few-Shot Learners},
      author={Chaofan Li and MingHao Qin and Shitao Xiao and Jianlyu Chen and Kun Luo and Yingxia Shao and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2409.15700},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.15700},
}
""",
)

bge_m3_unsupervised = ModelMeta(
    loader=sentence_transformers_loader,
    name="BAAI/bge-m3-unsupervised",
    languages=bgem3_languages,
    open_weights=True,
    revision="46f03bc86361cf88102b0b517b36c8259f2946b1",
    release_date="2024-01-30",  # January 30, 2024 - BGE-M3 release date
    n_parameters=568_000_000,
    memory_usage_mb=2167,
    embed_dim=1024,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/BAAI/bge-m3-unsupervised",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/FlagOpen/FlagEmbedding",
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_m3_training_data,
)

manu__bge_m3_custom_fr = ModelMeta(
    name="manu/bge-m3-custom-fr",
    revision="ed3ef88678ba83ddf4c0fab71a93cb90d89a9078",
    release_date="2024-04-11",
    languages=None,
    loader=sentence_transformers_loader,
    n_parameters=567754752,
    memory_usage_mb=2166,
    max_tokens=8194.0,
    embed_dim=1024,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/manu/bge-m3-custom-fr",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=bge_m3_training_data,
    adapted_from="BAAI/bge-m3",
    superseded_by=None,
)
