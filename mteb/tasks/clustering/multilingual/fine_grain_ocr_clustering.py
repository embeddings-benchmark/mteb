from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class FineGrainOCRITClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FineGrainOCRITClustering",
        description=(
            "Clustering grocery products from paired checkout images and Google "
            "Vision OCR text. The usable FineGrainOCR validation split contains "
            "18,389 examples from 256 barcode-registered product classes; 27 "
            "source rows with empty OCR are excluded. Barcode-like digit "
            "sequences are redacted from the OCR text to prevent label leakage."
        ),
        reference="https://doi.org/10.1007/s00138-024-01549-9",
        dataset={
            "path": "pranitchawla/FineGrainOCRITClustering",
            "revision": "02e86cf4e8dbeb0811ea5e889f7359eb7ae1b5f5",
        },
        type="ImageClustering",
        category="it2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn", "fra-Latn"],
        main_score="v_measure",
        date=("2019-02-17", "2019-09-30"),
        domains=["E-commerce"],
        task_subtypes=["Object recognition"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="found",
        prompt=(
            "Identify the grocery product shown in the image and described by "
            "the OCR text."
        ),
        bibtex_citation=r"""
@article{pettersson2024,
  title = {Multimodal fine-grained grocery product recognition using image and OCR text},
  author = {Pettersson, Tobias and Riveiro, Maria and L{\"o}fstr{\"o}m, Tuwe},
  journal = {Machine Vision and Applications},
  volume = {35},
  number = {4},
  pages = {79},
  year = {2024},
  publisher = {Springer},
  doi = {10.1007/s00138-024-01549-9},
}
""",
        is_beta=True,
    )

    max_fraction_of_documents_to_embed = None
    input_column_name = ("image", "text")
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["image", "text", "label"],
            )
