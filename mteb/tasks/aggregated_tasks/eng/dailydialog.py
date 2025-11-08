from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks import DailyDialogClassificationAct, DailyDialogClassificationEmotion


class DailyDialogClassification(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="DailyDialogClassification",
        description="",
        reference="",
        tasks=[
            DailyDialogClassificationAct(), DailyDialogClassificationEmotion(),
        ],
        category="t2c",
        license="not specified",
        modalities=["text"],
        annotations_creators=None,
        dialect=[""],
        eval_splits=["test", "validation"],
        eval_langs=["eng-Latn"],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="Classification",
        bibtex_citation=None,
    )
