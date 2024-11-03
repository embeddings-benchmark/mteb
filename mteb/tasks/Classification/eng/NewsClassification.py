from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NewsClassification",
        description="Large News Classification Dataset",
        dataset={
            "path": "fancyzhx/ag_news",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        },
        reference="https://arxiv.org/abs/1509.01626",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2004-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of news articles
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=["eng-Latn-US", "en-Latn-GB", "en-Latn-AU"],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{NIPS2015_250cf8b5,
        author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
        pages = {},
        publisher = {Curran Associates, Inc.},
        title = {Character-level Convolutional Networks for Text Classification},
        url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
        volume = {28},
        year = {2015}
        }""",
    )
