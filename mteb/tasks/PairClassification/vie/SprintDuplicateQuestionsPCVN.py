from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SprintDuplicateQuestionsPCVN(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SprintDuplicateQuestions-VN",
        description="Duplicate questions from the Sprint community.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "GreenNode/sprintduplicatequestions-pairclassification-vn",
            "revision": "2552beae0e4fe7fe05d088814f78a4c309ad2219",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["vie-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
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
        n_samples={"validation": 101000, "test": 101000},
        avg_character_length={"validation": 65.2, "test": 67.9},
    )
