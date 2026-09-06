from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SciRepEvalMeSHDescriptorsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SciRepEvalMeSHDescriptorsClassification",
        description=(
            "MeSH descriptor classification. Given the title and abstract of a "
            "biomedical paper, predict its Medical Subject Headings (MeSH) "
            "descriptor from a set of 30 common disease-related descriptors "
            "(multi-class classification). Published as a part of SciRepEval."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.338/",
        dataset={
            "path": "allenai/scirepeval",
            "name": "mesh_descriptors",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "split": "evaluation",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["evaluation"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-01", "2023-12-06"),
        domains=["Academic", "Medical", "Written"],
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

    is_cross_validation: bool = True
    train_split: str = "evaluation"

    def dataset_transform(self, num_proc: int | None = None) -> None:
        # Only the "evaluation" split is loaded (see `split` in metadata.dataset).
        # Each row is a (paper, descriptor) pair; deduplicate by paper so a single
        # abstract cannot appear in multiple cross-validation folds with different
        # labels. The deduplicated split is kept named "evaluation" and
        # MTEB's cross-validation path (is_cross_validation) runs KFold over it.
        from datasets import DatasetDict

        ds = self.dataset
        seen: set = set()

        def _first_per_paper(example: dict) -> bool:
            cid = example["corpus_id"]
            if cid in seen:
                return False
            seen.add(cid)
            return True

        ds = ds.filter(_first_per_paper)
        ds = ds.map(
            lambda x: {
                "text": f"{x['title'] or ''}\n\n{x['abstract'] or ''}".strip(),
                "label": x["descriptor"],
            },
            num_proc=num_proc,
        )
        keep = {"text", "label"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds = ds.class_encode_column("label")
        ds = ds.shuffle(seed=self.seed).select(range(min(len(ds), 8192)))
        self.dataset = DatasetDict({"evaluation": ds})
