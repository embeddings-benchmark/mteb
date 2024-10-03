from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification",
            "revision": "897743346c7c794264f7dbfadc3978aa2895e8e2",
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
