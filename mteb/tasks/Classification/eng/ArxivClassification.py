from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ArxivClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ArxivClassification",
        description="Classification Dataset of Arxiv Papers",
        dataset={
            "path": "ccdv/arxiv-classification",
            "revision": "f9bd92144ed76200d6eb3ce73a8bd4eba9ffdc85",
        },
        reference="https://ieeexplore.ieee.org/document/8675939",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(1998-11-11, 2019-3-28),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@ARTICLE{8675939,
        author={He, Jun and Wang, Liqun and Liu, Liu and Feng, Jiao and Wu, Hao},
        journal={IEEE Access}, 
        title={Long Document Classification From Local Word Glimpses via Recurrent Attention Learning}, 
        year={2019},
        volume={7},
        number={},
        pages={40707-40718},
        doi={10.1109/ACCESS.2019.2907992}
        }""",
        n_samples={"test": 2500},
        avg_character_length=None,
    )
