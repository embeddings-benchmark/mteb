from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact",
        dataset={
            "path": "mteb/scifact",
            "revision": "d56462d0e63a25450459c4f213e49ffdb866f7f9",
        },
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Medical", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
    )
