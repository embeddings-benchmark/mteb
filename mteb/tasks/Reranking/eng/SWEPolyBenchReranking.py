from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEPolyBenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEPolyBenchRR",
        description="Multilingual Software Issue Localization.",
        reference="https://amazon-science.github.io/SWE-PolyBench/",
        dataset={
            "path": "mteb/SWEPolyBenchRR",
            "revision": "3b896b59fb5892ab07a34a00917884f8690eaf9c",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2023-10-10", "2023-10-10"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a github issue, identify the code that needs to be changed to fix the issue."
        },
        bibtex_citation=r"""
@misc{rashid2025swepolybenchmultilanguagebenchmarkrepository,
  archiveprefix = {arXiv},
  author = {Muhammad Shihab Rashid and Christian Bock and Yuan Zhuang and Alexander Buchholz and Tim Esler and Simon Valentin and Luca Franceschi and Martin Wistuba and Prabhu Teja Sivaprasad and Woo Jung Kim and Anoop Deoras and Giovanni Zappella and Laurent Callot},
  eprint = {2504.08703},
  primaryclass = {cs.SE},
  title = {SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents},
  url = {https://arxiv.org/abs/2504.08703},
  year = {2025},
}
""",
    )
