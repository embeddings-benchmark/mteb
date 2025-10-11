from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class PUGGRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PUGGRetrieval",
        description="Information Retrieval PUGG dataset for the Polish language.",
        reference="https://aclanthology.org/2024.findings-acl.652/",
        dataset={
            "path": "clarin-pl/PUGG_IR",
            "revision": "48eff464950391ce7a3d58f37355fceccf613725",
        },
        type="Retrieval",
        category="t2t",
        date=("2023-01-01", "2024-01-01"),
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=r"""
@inproceedings{sawczyn-etal-2024-developing,
  address = {Bangkok, Thailand},
  author = {Sawczyn, Albert  and
Viarenich, Katsiaryna  and
Wojtasik, Konrad  and
Domoga{\l}a, Aleksandra  and
Oleksy, Marcin  and
Piasecki, Maciej  and
Kajdanowicz, Tomasz},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  doi = {10.18653/v1/2024.findings-acl.652},
  editor = {Ku, Lun-Wei  and
Martins, Andre  and
Srikumar, Vivek},
  month = aug,
  pages = {10978--10996},
  publisher = {Association for Computational Linguistics},
  title = {Developing {PUGG} for {P}olish: A Modern Approach to {KBQA}, {MRC}, and {IR} Dataset Construction},
  url = {https://aclanthology.org/2024.findings-acl.652/},
  year = {2024},
}
""",
    )
