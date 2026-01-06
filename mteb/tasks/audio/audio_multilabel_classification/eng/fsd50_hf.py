from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mteb.abstasks import AbsTaskMultilabelClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FSD50HFMultilingualClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="FSD50K",
        description="Multilabel Audio Classification on a subsampled version of FSD50K using 2048 samples",
        reference="https://huggingface.co/datasets/mteb/fsd50k_mini",
        dataset={
            "path": "mteb/fsd50k_mini",
            "revision": "a574eeb10bb11d28eff83dd522151028d529551d",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-01-30",
        ),
        domains=["Web"],  # obtained from Freesound - online collaborative platform
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{9645159,
  author = {Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  doi = {10.1109/TASLP.2021.3133208},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  keywords = {Videos;Task analysis;Labeling;Vocabulary;Speech recognition;Ontologies;Benchmark testing;Audio dataset;sound event;recognition;classification;tagging;data collection;environmental sound},
  number = {},
  pages = {829-852},
  title = {FSD50K: An Open Dataset of Human-Labeled Sound Events},
  volume = {30},
  year = {2022},
}
""",
    )

    evaluator = MultiOutputClassifier(estimator=LogisticRegression())
    input_column_name: str = "audio"
    label_column_name: str = "labels"
    samples_per_label: int = 8

