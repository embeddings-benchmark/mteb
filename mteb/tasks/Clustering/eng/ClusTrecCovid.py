from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
)
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class ClusTrecCovid(AbsTaskClusteringFast, MultilingualTask):
    metadata = TaskMetadata(
        name="ClusTREC-Covid",
        description="A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.",
        reference="https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID",
        dataset={
            "path": "Uri-ka/ClusTREC-Covid",
            "revision": "7f3489153b8dad7336a54f63202deb1414c33309",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"title and abstract": ["eng-Latn"], "title": ["eng-Latn"]},
        main_score="v_measure",
        date=("2020-04-10", "2020-07-16"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{katz-etal-2024-knowledge,
  address = {Miami, Florida, USA},
  author = {Katz, Uri  and
Levy, Mosh  and
Goldberg, Yoav},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  month = nov,
  pages = {8838--8855},
  publisher = {Association for Computational Linguistics},
  title = {Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature},
  url = {https://aclanthology.org/2024.findings-emnlp.516},
  year = {2024},
}
""",
        prompt="Identify the main category of the covid-19 papers based on the titles and abstracts",
    )
