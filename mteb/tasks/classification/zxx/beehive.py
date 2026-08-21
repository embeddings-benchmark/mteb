from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class BeehiveStatesClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BeehiveStatesClassification",
        description=(
            "Classify ten-minute beehive recordings as queen-present or queenless "
            "across two hive-independent folds."
        ),
        reference="https://arxiv.org/abs/1811.06330",
        dataset={
            "path": "artist/BeehiveStatesClassification",
            "revision": "2f920226b3153596c01c52fb5b25e5c26c516fae",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
        eval_langs={"fold0": ["zxx-Zxxx"], "fold1": ["zxx-Zxxx"]},
        # HEAR reports AUROC and top-1 accuracy. The generic MTEB classification
        # evaluator does not expose AUROC, so use the metric shared by both.
        main_score="accuracy",
        date=("2017-07-14", "2018-06-12"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{nolasco2019beehive,
  author = {Ines Nolasco and Alessandro Terenzi and Stefania Cecchi and Simone Orcioni and Helen L. Bear and Emmanouil Benetos},
  booktitle = {2019 IEEE International Conference on Acoustics, Speech and Signal Processing},
  doi = {10.1109/ICASSP.2019.8682981},
  pages = {8256--8260},
  title = {Audio-based Identification of Beehive States},
  year = {2019},
}
""",
        is_beta=True,
    )

    input_column_name = "audio"
    label_column_name = "label"
