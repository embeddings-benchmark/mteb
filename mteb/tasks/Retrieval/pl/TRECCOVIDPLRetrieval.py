from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TRECCOVIDPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-PL",
        description="TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        hf_hub_name="clarin-knext/trec-covid-pl",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ndcg_at_10",
        revision="81bcb408f33366c2a20ac54adafad1ae7e877fdd",
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
        n_samples={},
        avg_character_length={},
    )
