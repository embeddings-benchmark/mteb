from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TRECCOVID(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID",
        description="TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        hf_hub_name="mteb/trec-covid",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        revision="1271c7809071a13532e05f25fb53511ffce77117",
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
