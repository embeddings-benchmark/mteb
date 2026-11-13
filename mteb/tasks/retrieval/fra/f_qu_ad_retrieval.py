from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FQuADRetrieval",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "mteb/FQuADRetrieval",
            "revision": "cba2a9a9bd2183a387baa99555811d65d2abbf8e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2019-11-01", "2020-05-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{dhoffschmidt-etal-2020-fquad,
  address = {Online},
  author = {d{'}Hoffschmidt, Martin  and
Belblidia, Wacim  and
Heinrich, Quentin  and
Brendl{\'e}, Tom  and
Vidal, Maxime},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  doi = {10.18653/v1/2020.findings-emnlp.107},
  editor = {Cohn, Trevor  and
He, Yulan  and
Liu, Yang},
  month = nov,
  pages = {1193--1208},
  publisher = {Association for Computational Linguistics},
  title = {{FQ}u{AD}: {F}rench Question Answering Dataset},
  url = {https://aclanthology.org/2020.findings-emnlp.107},
  year = {2020},
}
""",
    )
