from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SyntecReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SyntecReranking",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        hf_hub_name="lyon-nlp/mteb-fr-reranking-syntec-s2p",
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="map",
        revision="b205c5084a0934ce8af14338bf03feb19499c84d",
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
