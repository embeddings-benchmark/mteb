from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FSD50KA2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FSD50KA2ARetrieval",
        description=(
            "Audio-to-audio sound-event retrieval on FSD50K. Queries and corpus "
            "are disjoint Freesound recordings drawn from FSD50K's public eval "
            "split, sampled across 20 well-populated AudioSet-ontology sound-event "
            "classes (100 queries / 200 corpus docs, 5 queries and 10 corpus docs "
            "per class); relevance is same-class membership using each clip's "
            "primary (most specific) label."
        ),
        reference="https://arxiv.org/abs/2010.00475",
        dataset={
            "path": "yaswanth169/FSD50K-A2ARetrieval",
            "revision": "c682e88dcc565ca525b918f8f440c25ad49329cf",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2015-01-01", "2020-10-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{fonseca2022FSD50K,
  author = {Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  pages = {829--852},
  title = {{FSD50K}: an Open Dataset of Human-Labeled Sound Events},
  volume = {30},
  year = {2022},
}
""",
        prompt={"query": "Retrieve other recordings of the same sound event category."},
        is_beta=True,
    )
