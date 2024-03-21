from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SprintDuplicateQuestionsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SprintDuplicateQuestions",
        description="Duplicate questions from the Sprint community.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        hf_hub_name="mteb/sprintduplicatequestions-pairclassification",
        type="PairClassification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["en"],
        main_score="ap",
        revision="d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46",
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
    )
