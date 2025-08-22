from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFactPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-PL",
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        dataset={
            "path": "clarin-knext/scifact-pl",
            "revision": "47932a35f045ef8ed01ba82bf9ff67f6e109207e",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Medical", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{wojtasik2024beirpl,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  year = {2024},
}
""",
        adapted_from=["SciFact"],
    )
