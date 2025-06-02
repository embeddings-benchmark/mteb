from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MMMARCONL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="mMARCO-NL",
        dataset={
            "path": "clips/beir-nl-mmarco",
            "revision": "4a6c6c38794088dd4e227a9fe3595a3d188ccf95",
        },
        description="mMARCO is a multi-lingual (translated) collection of datasets focused on deep learning in search",
        reference="https://github.com/unicamp-dl/mMARCO",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-30"),  # best guess: based on publication date
        domains=["Web", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="derived",  # manually checked a small subset
        dialect=[],
        sample_creation="machine-translated and verified",
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
    )
