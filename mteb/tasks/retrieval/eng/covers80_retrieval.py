from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Covers80A2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Covers80A2ARetrieval",
        description=(
            "Audio-to-audio cover-song retrieval on the classic Covers80 dataset: "
            "80 songs with 2 recordings each (160 tracks). A query recording should "
            "retrieve the other recording of the same work (cover / alternate "
            "performance). Packaged from the LabROSA covers dataset."
        ),
        reference="https://labrosa.ee.columbia.edu/projects/coversongs/covers80/",
        dataset={
            "path": "Wissam42/Covers80-A2A",
            "revision": "37145bdba45d5593c8a171998dd797a360c6cf96",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2007-01-01", "2007-12-31"),
        domains=["Music"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{ellis2007covers80,
  author = {Daniel P. W. Ellis and Brian Whitman},
  title = {The Covers80 Cover Song Dataset},
  url = {https://labrosa.ee.columbia.edu/projects/coversongs/covers80/},
  year = {2007},
}
""",
        prompt={
            "query": "Retrieve another recording (cover) of the same musical work."
        },
        is_beta=True,
    )
