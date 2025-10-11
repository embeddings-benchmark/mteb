from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RP2kI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RP2kI2IRetrieval",
        description="Retrieve photos of 39457 products.",
        reference="https://arxiv.org/abs/2006.12634",
        dataset={
            "path": "JamieSJS/rp2k",
            "revision": "f8f82d4eb1aa4dc4dbf2c768596c8110a3703765",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2009-01-01", "2010-04-01"),
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@article{peng2020rp2k,
  author = {Peng, Jingtian and Xiao, Chang and Li, Yifan},
  journal = {arXiv preprint arXiv:2006.12634},
  title = {RP2K: A large-scale retail product dataset for fine-grained image classification},
  year = {2020},
}
""",
    )
    skip_first_result = True
