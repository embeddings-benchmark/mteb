from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Country211Classification(AbsTaskClassification):
    input_column_name: str = "jpg"
    label_column_name: str = "cls"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="Country211",
        description="Classifying images of 211 countries.",
        reference="https://huggingface.co/datasets/clip-benchmark/wds_country211",
        dataset={
            "path": "mteb/wds_country211",
            "revision": "c3875324cab4803fa6d3afcd5947a97c84fad4c6",
        },
        type="ImageClassification",
        category="i2c",
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
        modalities=["image"],
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
