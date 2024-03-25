from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class AlloprofReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AlloprofReranking",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        hf_hub_name="mteb/alloprof-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="map",
        revision="666fdacebe0291776e86f29345663dfaf80a0db9",
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
