from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Ocnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Ocnli",
        description="Original Chinese Natural Language Inference dataset",
        reference="https://arxiv.org/abs/2010.05444",
        dataset={
            "path": "C-MTEB/OCNLI",
            "revision": "66e76a618a34d6d565d5538088562851e6daa7ec",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
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
        n_samples=None,
        avg_character_length=None,
    )


class Cmnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Cmnli",
        description="Chinese Multi-Genre NLI",
        reference="https://huggingface.co/datasets/clue/viewer/cmnli",
        dataset={
            "path": "C-MTEB/CMNLI",
            "revision": "41bc36f332156f7adc9e38f53777c959b2ae9766",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
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
        n_samples=None,
        avg_character_length=None,
    )
