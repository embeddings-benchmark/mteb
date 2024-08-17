from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaHard2IsotopesVsFissionProductsNuclearFission(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaHard2IsotopesVsFissionProductsNuclearFission",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Hard_2_Class_Isotopes_vs_Fission_Products_Nuclear_Fission",
            "revision": "9c3974e039e774828742e739a3cc7bced7b337d5",
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
