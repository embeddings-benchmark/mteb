from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DanFeverRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DanFeverRetrieval",
        dataset={
            "path": "mteb/DanFeverRetrieval",
            "revision": "7339a3c35ef27a0bf2af829597bd7d297b69f81c",
        },
        description="A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.",
        reference="https://aclanthology.org/2021.nodalida-main.47/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{norregaard-derczynski-2021-danfever,
  address = {Reykjavik, Iceland (Online)},
  author = {N{\o}rregaard, Jeppe  and
Derczynski, Leon},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {422--428},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {{D}an{FEVER}: claim verification dataset for {D}anish},
  url = {https://aclanthology.org/2021.nodalida-main.47},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim in Danish, retrieve documents that support the claim"
        },
        task_subtypes=["Claim verification"],
    )


class DanFever(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="DanFEVER",
        dataset={
            "path": "mteb/DanFEVER",
            "revision": "5222ad7225b04f192be495bbd83b198569eef810",
        },
        description="A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.",
        reference="https://aclanthology.org/2021.nodalida-main.47/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{norregaard-derczynski-2021-danfever,
  address = {Reykjavik, Iceland (Online)},
  author = {N{\o}rregaard, Jeppe  and
Derczynski, Leon},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {422--428},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {{D}an{FEVER}: claim verification dataset for {D}anish},
  url = {https://aclanthology.org/2021.nodalida-main.47},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim in Danish, retrieve documents that support the claim"
        },
        task_subtypes=["Claim verification"],
        superseded_by="DanFeverRetrieval",
    )
