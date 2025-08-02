from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterSemEval2015PCVN(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterSemEval2015-VN",
        dataset={
            "path": "GreenNode/twittersemeval2015-pairclassification-vn",
            "revision": "9215a3c954078fd15c2bbecca914477d53944de1",
        },
        description="Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
        reference="https://alt.qcri.org/semeval2015/task1/",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ap",
        date=("2025-07-29", "2025-07-30"),
        form=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        socioeconomic_status=None,
        text_creation=None,
        bibtex_citation="""
@misc{pham2025vnmtebvietnamesemassivetext,
    title={VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
    author={Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
    year={2025},
    eprint={2507.21500},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.21500}
}
""",
        adapted_from=["TwitterSemEval2015"],
    )
