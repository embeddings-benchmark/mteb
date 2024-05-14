from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Reranking.zho.CMTEBReranking import CMedQAv2, T2Reranking


class T2RerankingAbstention(AbsTaskAbstention, T2Reranking):
    abstention_task = "Reranking"
    metadata = TaskMetadata(
        name="T2Reranking",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "C-MTEB/T2Reranking",
            "revision": "76631901a18387f85eaa53e5450019b87ad58ef9",
        },
        type="Abstention",
        category="s2s",
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
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


class CMedQAv2Abstention(AbsTaskAbstention, CMedQAv2):
    abstention_task = "Reranking"
    metadata = TaskMetadata(
        name="CMedQAv2-rerankingAbstention",
        description="Chinese community medical question answering",
        reference="https://github.com/zhangsheng93/cMedQA2",
        dataset={
            "path": "C-MTEB/CMedQAv2-reranking",
            "revision": "23d186750531a14a0357ca22cd92d712fd512ea0",
        },
        type="Abstention",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
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
