from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalSummarization(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalSummarization",
        description="The dataset consistes of 439 pairs of contracts and their summarizations from https://tldrlegal.com and https://tosdr.org/.",
        reference="https://github.com/lauramanor/legal_summarization",
        dataset={
            "path": "mteb/legal_summarization",
            "revision": "3bb1a05c66872889662af04c5691c14489cebd72",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["en"],
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