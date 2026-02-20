from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BSARDRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="BSARDRetrieval",
        description="The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.",
        reference="https://huggingface.co/datasets/maastrichtlawtech/bsard",
        dataset={
            "path": "mteb/BSARDRetrieval",
            "revision": "8c492add6a14ac188f2debdaf6cbdfb406fd6be3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="recall_at_100",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Spoken"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{louis2022statutory,
  address = {Dublin, Ireland},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2022.acl-long.468},
  month = may,
  pages = {6789–6803},
  publisher = {Association for Computational Linguistics},
  title = {A Statutory Article Retrieval Dataset in French},
  url = {https://aclanthology.org/2022.acl-long.468/},
  year = {2022},
}
""",
        superseded_by="BSARDRetrieval.v2",
    )


class BSARDRetrievalv2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="BSARDRetrieval.v2",
        description="BSARD is a French native dataset for legal information retrieval. BSARDRetrieval.v2 covers multi-article queries, fixing issues (#2906) with the previous data loading. ",
        reference="https://huggingface.co/datasets/maastrichtlawtech/bsard",
        dataset={
            "path": "mteb/BSARDRetrieval.v2",
            "revision": "e4b6c883c5bb602e1ac46d2939484ff40b1545f4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="recall_at_100",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Spoken"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{louis2022statutory,
  address = {Dublin, Ireland},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2022.acl-long.468},
  month = may,
  pages = {6789–6803},
  publisher = {Association for Computational Linguistics},
  title = {A Statutory Article Retrieval Dataset in French},
  url = {https://aclanthology.org/2022.acl-long.468/},
  year = {2022},
}
""",
    )
