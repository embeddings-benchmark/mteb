from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class StackOverflowDupQuestions(AbsTaskReranking):
    metadata = TaskMetadata(
        name="StackOverflowDupQuestions",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "mteb/stackoverflowdupquestions-reranking",
            "revision": "e185fbe320c72810689fc5848eb6114e1ef5ec69",
        },
        type="Reranking",
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
        bibtex_citation="""@article{Liu2018LinkSOAD,
  title={LinkSO: a dataset for learning to retrieve similar question answer pairs on software development forums},
  author={Xueqing Liu and Chi Wang and Yue Leng and ChengXiang Zhai},
  journal={Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering},
  year={2018},
  url={https://api.semanticscholar.org/CorpusID:53111679}
}""",
        n_samples={"test": 3467},
        avg_character_length={"test": 49.8},
    )
