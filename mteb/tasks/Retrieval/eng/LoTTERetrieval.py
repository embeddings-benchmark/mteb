from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class LoTTERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "url": "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz",
            "path": "colbertv2/lotte",
            "revision": "main",
        },
        name="LoTTE",
        description=(
            "LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models "
            "on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and "
            "passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing "
            "a challenging out-of-domain generalization benchmark."
        ),
        type="Retrieval",
        modalities=["text"],
        category="s2s",
        reference="https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        eval_langs_per_domain={
            "writing": ["eng-Latn"],
            "recreation": ["eng-Latn"],
            "science": ["eng-Latn"],
            "technology": ["eng-Latn"],
            "lifestyle": ["eng-Latn"],
        },
        main_score="success@5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Academic", "Web", "Social"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{santhanam2021colbertv2,
            title={ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction},
            author={Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and Potts, Christopher and Zaharia, Matei},
            journal={arXiv preprint arXiv:2112.01488},
            year={2021}
        }""",
        prompt=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_loaded = False

    def dataset_transform(self, data: dict) -> dict:
        # Merge the nested dictionaries for queries and relevant docs,
        # while leaving corpus as-is.
        split = self.metadata.eval_splits[0]
        return {
            split: {
                "queries": {
                    k: v for d in data["queries"][split].values() for k, v in d.items()
                },
                "corpus": data["corpus"][split],
                "relevant": {
                    k: v
                    for d in data["relevant_docs"][split].values()
                    for k, v in d.items()
                },
            }
        }
