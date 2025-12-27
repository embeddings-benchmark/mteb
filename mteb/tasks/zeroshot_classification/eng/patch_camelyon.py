from pathlib import Path

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class PatchCamelyonZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="PatchCamelyonZeroShot",
        description="Histopathology diagnosis classification dataset.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24",
        dataset={
            "path": "clip-benchmark/wds_vtab-pcam",
            "revision": "502695fe1a141108650e3c5b91c8b5e0ff84ed49",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2018-01-01",
            "2018-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Medical"],
        task_subtypes=["Tumor detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{10.1007/978-3-030-00934-2_24,
  address = {Cham},
  author = {Veeling, Bastiaan S.
and Linmans, Jasper
and Winkens, Jim
and Cohen, Taco
and Welling, Max},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2018},
  editor = {Frangi, Alejandro F.
and Schnabel, Julia A.
and Davatzikos, Christos
and Alberola-L{\'o}pez, Carlos
and Fichtinger, Gabor},
  isbn = {978-3-030-00934-2},
  pages = {210--218},
  publisher = {Springer International Publishing},
  title = {Rotation Equivariant CNNs for Digital Pathology},
  year = {2018},
}
""",
    )
    input_column_name = "webp"
    label_column_name = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = Path(__file__).parent / "templates" / "PatchCamelyon_labels.txt"
        with path.open() as f:
            labels = f.readlines()

        return [f"histopathology image of {c}" for c in labels]
