from __future__ import annotations

from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SciRepEvalFoSClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="SciRepEvalFoSClassification",
        description=(
            "Fields of Study (FoS) classification. Given the title and abstract "
            "of a scientific paper, predict one or more of its fields of study "
            "(multi-label classification). Published as a part of SciRepEval."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "name": "fos",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "split": "evaluation",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-12-06"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification"],
        license="odc-by",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{singh-etal-2023-scirepeval,
  address = {Singapore},
  author = {Singh, Amanpreet  and
D{'}Arcy, Mike  and
Cohan, Arman  and
Downey, Doug  and
Feldman, Sergey},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2023.emnlp-main.338},
  month = dec,
  pages = {5548--5566},
  publisher = {Association for Computational Linguistics},
  title = {{SciRepEval: A Multi-Format Benchmark for Scientific Document Representations}},
  url = {https://aclanthology.org/2023.emnlp-main.338/},
  year = {2023},
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        # Only the "evaluation" split is loaded (see `split` in metadata.dataset);
        # build text/label columns and a random train/test split for the probe.
        ds = self.dataset
        ds = ds.map(
            lambda x: {
                "text": f"{x['title'] or ''}\n\n{x['abstract'] or ''}".strip(),
                "label": x["labels_text"],
            },
            num_proc=num_proc,
        )
        ds = ds.filter(lambda x: len(x["label"]) > 0, num_proc=num_proc)
        keep = {"text", "label"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds = ds.shuffle(seed=self.seed).select(range(min(len(ds), 8192)))
        self.dataset = ds.train_test_split(test_size=0.3, seed=self.seed)
