from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.dialog_state_tracking.eng.multi_woz import (
    MultiWoz21Attraction,
    MultiWoz21Hotel,
    MultiWoz21Restaurant,
    MultiWoz21Train,
)


class MultiWoz21(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MultiWoz21",
        description="",
        reference="",
        tasks=[
            MultiWoz21Attraction(),
            # MultiWoz21Hospital(),
            MultiWoz21Hotel(),
            MultiWoz21Restaurant(),
            MultiWoz21Train(),
        ],
        category="t2c",
        license="not specified",
        modalities=["text"],
        annotations_creators=None,
        dialect=[""],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="Classification",
        bibtex_citation=None,
    )
