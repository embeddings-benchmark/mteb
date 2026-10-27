from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MultiLongDocReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiLongDocReranking",
        description=(
            "Reranking version of MultiLongDocRetrieval (MLDR). MLDR is a Multilingual Long-Document "
            "Retrieval dataset built on Wikipedia, Wudao and mC4, covering 13 typologically diverse languages. "
            "Specifically, we sample lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose "
            "paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. "
            "The generated question and the sampled article constitute a new text pair to the dataset."
        ),
        reference="https://huggingface.co/datasets/Shitao/MLDR",
        dataset={
            "path": "mteb/MultiLongDocReranking",
            "revision": "ad09ce14c17bce6edae151b7f6ef12e15d91dbf3",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "ar": ["ara-Arab"],
            "de": ["deu-Latn"],
            "en": ["eng-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "hi": ["hin-Deva"],
            "it": ["ita-Latn"],
            "ja": ["jpn-Jpan"],
            "ko": ["kor-Kore"],
            "pt": ["por-Latn"],
            "ru": ["rus-Cyrl"],
            "th": ["tha-Thai"],
            "zh": ["zho-Hans"],
        },
        main_score="ndcg_at_10",
        date=(
            "2000-01-01",
            "2024-12-31",
        ),  # Not found in the paper, guessed using the paper's publication date and constituent datasets
        domains=[
            "Encyclopaedic",
            "Written",
            "Web",
            "Non-fiction",
            "Fiction",
        ],  # narrativeqa, wikipedia, wudao, mC4
        task_subtypes=[],
        license="mit",
        annotations_creators="LM-generated",  # gpt-3.5
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{bge-m3,
  address = {Bangkok, Thailand},
  author = {Chen, Jianlyu and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2024},
  doi = {10.18653/v1/2024.findings-acl.137},
  editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  month = {aug},
  pages = {2318--2335},
  publisher = {Association for Computational Linguistics},
  title = {{M}3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  url = {https://aclanthology.org/2024.findings-acl.137/},
  year = {2024},
}
""",
        prompt={
            "query": "Given a question, rerank long documents based on their relevance to answer the question"
        },
        adapted_from=["MultiLongDocRetrieval"],
    )
