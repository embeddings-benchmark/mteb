from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PatchCamelyonClassification(AbsTaskClassification):
    input_column_name = "webp"
    label_column_name = "cls"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="PatchCamelyon",
        description="Histopathology diagnosis classification dataset.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24",
        dataset={
            "path": "clip-benchmark/wds_vtab-pcam",
            "revision": "502695fe1a141108650e3c5b91c8b5e0ff84ed49",
        },
        type="ImageClassification",
        category="i2c",
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
        modalities=["image"],
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
