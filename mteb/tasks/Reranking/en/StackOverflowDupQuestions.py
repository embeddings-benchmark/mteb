from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class StackOverflowDupQuestions(AbsTaskReranking):
    metadata = TaskMetadata(
        name="StackOverflowDupQuestions",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        hf_hub_name="mteb/stackoverflowdupquestions-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="map",
        revision="e185fbe320c72810689fc5848eb6114e1ef5ec69",
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

