from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SQuADKorV1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SQuADKorV1Retrieval",
        description="Korean translation of SQuAD v1.0 dataset for retrieval task, based on Korean Wikipedia articles.",
        reference="https://huggingface.co/datasets/yjoonjang/squad_kor_v1",
        dataset={
            "path": "yjoonjang/squad_kor_v1",
            "revision": "2b4ee1f3b143a04792da93a3df21933c5fe9eed3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{rajpurkar-etal-2016-squad,
  address = {Austin, Texas},
  author = {Rajpurkar, Pranav  and
Zhang, Jian  and
Lopyrev, Konstantin  and
Liang, Percy},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D16-1264},
  editor = {Su, Jian  and
Duh, Kevin  and
Carreras, Xavier},
  month = nov,
  pages = {2383--2392},
  publisher = {Association for Computational Linguistics},
  title = {{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text},
  url = {https://aclanthology.org/D16-1264},
  year = {2016},
}
""",
    )
