from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Reranking.eng.StackOverflowDupQuestions import \
    StackOverflowDupQuestions


class StackOverflowDupQuestionsAbstention(AbsTaskAbstention, StackOverflowDupQuestions):
    abstention_task = "Reranking"
    metadata = TaskMetadata(
        name="StackOverflowDupQuestionsAbstention",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "mteb/stackoverflowdupquestions-reranking",
            "revision": "e185fbe320c72810689fc5848eb6114e1ef5ec69",
        },
        type="Abstention",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples={"test": 3467},
        avg_character_length={"test": 49.8},
    )
