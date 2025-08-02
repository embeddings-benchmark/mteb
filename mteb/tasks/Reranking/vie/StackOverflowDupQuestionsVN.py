from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking


class StackOverflowDupQuestionsVN(AbsTaskReranking):
    metadata = TaskMetadata(
        name="StackOverflowDupQuestions-VN",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "GreenNode/stackoverflowdupquestions-reranking-vn",
            "revision": "3ceb17db245f52beaf27a3720aa71e1cc5f06faf",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="map",
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
        adapted_from=["StackOverflowDupQuestions"],
    )
