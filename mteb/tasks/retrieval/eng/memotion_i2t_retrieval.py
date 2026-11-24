from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MemotionI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MemotionI2TRetrieval",
        description="Retrieve captions based on memes.",
        reference="https://aclanthology.org/2020.semeval-1.99/",
        dataset={
            "path": "mteb/MemotionI2TRetrieval",
            "revision": "4b431fdea826fc498caf7a1d3c509acbebdb86fa",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sharma2020semeval,
  author = {Sharma, Chhavi and Bhageria, Deepesh and Scott, William and Pykl, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
  booktitle = {Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages = {759--773},
  title = {SemEval-2020 Task 8: Memotion Analysis-the Visuo-Lingual Metaphor!},
  year = {2020},
}
""",
    )
