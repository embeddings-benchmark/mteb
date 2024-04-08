from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ImdbClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ImdbClassification",
        description="Large Movie Review Dataset",
        dataset={
            "path": "mteb/imdb",
            "revision": "3d86128a09e091d6018b6d26cad27f2739fc2db7",
        },
        reference="http://www.aclweb.org/anthology/P11-1015",
        type="Classification",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 25000},
        avg_character_length={"test": 1293.8},
    )
