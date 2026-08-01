from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMarcoRetrievalMultilingual(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MMarcoRetrievalMultilingual",
        description="A multilingual version of the MS MARCO passage retrieval dataset.",
        reference="https://github.com/unicamp-dl/mMARCO",
        dataset={
            "path": "lopozz/MMarcoRetrieval",
            # Pin this to the final Hugging Face commit before opening a PR.
            "revision": None,
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs={
            "it": ["ita-Latn"],
            "es": ["spa-Latn"],
            "zh": ["cmn-Hans"],
        },
        main_score="ndcg_at_10",
        date=("2016-01-01", "2021-08-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-2108-13897,
  author = {Luiz Bonifacio and
Israel Campiotti and
Roberto de Alencar Lotufo and
Rodrigo Frassetto Nogueira},
  eprint = {2108.13897},
  eprinttype = {arXiv},
  journal = {CoRR},
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
        contributed_by="lopozz",
        is_beta=True,
    )
