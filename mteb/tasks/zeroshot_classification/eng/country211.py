from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Country211ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Country211ZeroShot",
        description="Classifying images of 211 countries.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_country211",
        dataset={
            "path": "mteb/Country211ZeroShot",
            "revision": "3f26e6143f15468ce4fd685d100dd8c31ec48ad6",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2021-02-26",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{radford2021learning,
  archiveprefix = {arXiv},
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  eprint = {2103.00020},
  pages = {8748--8763},
  publisher = {PMLR},
  series = {Proceedings of Machine Learning Research},
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  url = {https://proceedings.mlr.press/v139/radford21a.html},
  volume = {139},
  year = {2021},
}
""",
    )
