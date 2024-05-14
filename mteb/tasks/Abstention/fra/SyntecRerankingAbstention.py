from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Reranking.fra.SyntecReranking import SyntecReranking


class SyntecRerankingAbstention(AbsTaskAbstention, SyntecReranking):
    abstention_task = "Reranking"
    metadata = TaskMetadata(
        name="SyntecRerankingAbstention",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "revision": "b205c5084a0934ce8af14338bf03feb19499c84d",
        },
        type="Abstention",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
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
