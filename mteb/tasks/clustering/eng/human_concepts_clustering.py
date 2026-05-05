from __future__ import annotations

from datasets import DatasetDict

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@article{shani2025tokens,
  author = {Shani, Chen and Soffer, Liron and Jurafsky, Dan and LeCun, Yann and Shwartz-Ziv, Ravid},
  journal = {arXiv preprint arXiv:2505.17117},
  title = {From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning},
  year = {2025},
}
"""

_SUBDATASETS = ["Rosch1973", "Rosch1975", "McCloskey1978"]


class HumanConceptsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="HumanConceptsClustering",
        description=(
            "Semantic concept clustering benchmark drawn from three classic cognitive psychology "
            "typicality studies: Rosch (1973, 48 items, 8 categories), Rosch (1975, 565 items, "
            "10 categories), and McCloskey & Glucksberg (1978, 492 items, 18 categories). "
            "Each item is an everyday concept (e.g. 'robin', 'chair') annotated with a "
            "human-assigned semantic category. The task measures how well embeddings recover "
            "human conceptual organisation. Evaluated separately per study."
        ),
        reference="https://arxiv.org/abs/2505.17117",
        dataset={
            "path": "CShani/human-concepts",
            "revision": "99f1715a5945341896ed3cbe4a3f1d3b7b913cf2",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=_SUBDATASETS,
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1973-01-01", "2025-05-22"),
        domains=["Academic"],
        task_subtypes=["Thematic clustering"],
        license="http://www.wtfpl.net/about/",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=CITATION,
        prompt="Identify the semantic category of the concept",
    )

    max_fraction_of_documents_to_embed = None
    input_column_name = "item"
    label_column_name = "category"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        self.dataset = DatasetDict({
            sub: self.dataset["train"].filter(
                lambda x, s=sub: x["subdataset"] == s,
                num_proc=num_proc,
            )
            for sub in _SUBDATASETS
        })
