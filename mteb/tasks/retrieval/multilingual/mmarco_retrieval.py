from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMarcoRetrievalMultilingual(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MMarcoRetrievalMultilingual",
        description="A retrieval task derived from mMARCO v2. 100,000 candidates were sampled from the entire corpus to reduce evaluation inference cost.",
        reference="https://github.com/unicamp-dl/mMARCO",
        dataset={
            "path": "lopozz/MMarcoRetrievalMultilingual",
            "revision": "44480423f56a42b948edd899c707e640b8e46f19",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs={
            "ar": ["ara-Arab"],
            "de": ["deu-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "hi": ["hin-Deva"],
            "id": ["ind-Latn"],
            "it": ["ita-Latn"],
            "ja": ["jpn-Jpan"],
            "nl": ["nld-Latn"],
            "pt": ["por-Latn"],
            "ru": ["rus-Cyrl"],
            "vi": ["vie-Latn"],
            "zh": ["cmn-Hans"],
        },
        main_score="ndcg_at_10",
        date=("2016-01-01", "2021-08-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-2108-13897,
  author = {Luiz Bonifacio and
Israel Campiotti and
Roberto de Alencar Lotufo and
Rodrigo Frassetto Nogueira},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-2108-13897.bib},
  eprint = {2108.13897},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Mon, 20 Mar 2023 15:35:34 +0100},
  title = {mMARCO: {A} Multilingual Version of {MS} {MARCO} Passage Ranking Dataset},
  url = {https://arxiv.org/abs/2108.13897},
  volume = {abs/2108.13897},
  year = {2021},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
        adapted_from=["MMarcoRetrieval"],
    )
