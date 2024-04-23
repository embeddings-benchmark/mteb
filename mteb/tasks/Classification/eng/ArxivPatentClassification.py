from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ArxivPatentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ArxivPatentClassification",
        description="Classification Dataset of Patents and Abstract",
        dataset={
            "path": "ccdv/patent-classification",
            "revision": "2f38a1dfdecfacee0184d74eaeafd3c0fb49d2a6",
        },
        reference="https://aclanthology.org/P19-1212.pdf",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=[],
        task_subtypes=None,
        license="Not specified",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=[],
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 2048},
        avg_character_length=None,
    )
