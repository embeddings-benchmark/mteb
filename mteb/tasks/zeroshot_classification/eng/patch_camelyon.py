from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class PatchCamelyonZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PatchCamelyonZeroShot",
        description="Histopathology diagnosis classification dataset.",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24",
        dataset={
            "path": "mteb/PatchCamelyonZeroShot",
            "revision": "b8966c8bba47b7a37d5e82b32855e06c4785cc2f",
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
