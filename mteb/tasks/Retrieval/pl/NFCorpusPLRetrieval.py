from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpusPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-PL",
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        dataset={
            "path": "clarin-knext/nfcorpus-pl",
            "revision": "9a6f9567fda928260afed2de480d79c98bf0bec0",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ndcg_at_10",
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
