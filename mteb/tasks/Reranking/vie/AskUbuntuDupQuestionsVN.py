from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking


class AskUbuntuDupQuestionsVN(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AskUbuntuDupQuestions-VN",
        description="AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "GreenNode/askubuntudupquestions-reranking-vn",
            "revision": "5cfaa5c07252d30c37302bfc056f0d85884971a1",
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
        adapted_from=["AskUbuntuDupQuestions"],
    )
