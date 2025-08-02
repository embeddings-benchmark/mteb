from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterURLCorpus-VN",
        dataset={
            "path": "GreenNode/twitterurlcorpus-pairclassification-vn",
            "revision": "6e6a40aaade2129f70432f2156a6d24b63d72be3",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
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
        adapted_from=["TwitterURLCorpus"],
    )
