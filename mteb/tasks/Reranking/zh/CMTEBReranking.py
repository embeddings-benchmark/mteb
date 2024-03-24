from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class T2Reranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="T2Reranking",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        hf_hub_name="C-MTEB/T2Reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["dev"],
        eval_langs=["zh"],
        main_score="map",
        revision="76631901a18387f85eaa53e5450019b87ad58ef9",
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


class MMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MMarcoReranking",
        description="mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
        reference="https://github.com/unicamp-dl/mMARCO",
        hf_hub_name="C-MTEB/Mmarco-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["dev"],
        eval_langs=["zh"],
        main_score="map",
        revision="8e0c766dbe9e16e1d221116a3f36795fbade07f6",
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


class CMedQAv1(AbsTaskReranking):
    metadata = TaskMetadata(
        name="CMedQAv1-reranking",
        description="Chinese community medical question answering",
        reference="https://github.com/zhangsheng93/cMedQA",
        hf_hub_name="C-MTEB/CMedQAv1-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="map",
        revision="cd540c506dae1cf9e9a59c3e06f42030d54e7301",
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


class CMedQAv2(AbsTaskReranking):
    metadata = TaskMetadata(
        name="CMedQAv2-reranking",
        description="Chinese community medical question answering",
        reference="https://github.com/zhangsheng93/cMedQA2",
        hf_hub_name="C-MTEB/CMedQAv2-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="map",
        revision="23d186750531a14a0357ca22cd92d712fd512ea0",
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
