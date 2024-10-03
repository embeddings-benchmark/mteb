from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2ComputationalVsSpectroscopistsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2ComputationalVsSpectroscopistsClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaMedium2ComputationalVsSpectroscopistsClassification",
            "revision": "474d706a22b0451b5846d623aa4b4234ba5b0513",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={}
    )
