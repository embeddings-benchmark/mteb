from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class IncompeBenchStrictRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IncompeBenchStrictRetrieval",
        description=(
            "IncompeBench-Strict is a fine-grained text-to-music retrieval "
            "benchmark with 500 diverse search queries, 1,574 permissively "
            "licensed music snippets, and graded relevance judgements. The "
            "two variants share the same queries and corpus; unlike Lenient, "
            "Strict drops score-1 tangential matches and retains only scores 2-3."
        ),
        reference="https://arxiv.org/abs/2602.11941",
        dataset={
            "path": "mteb/IncompeBenchStrictRetrieval",
            "revision": "1185590f69aa460a9d61164f66df920fda7f0013",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-29", "2026-02-12"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{clavie2026incompebench,
  archiveprefix = {arXiv},
  author = {Benjamin Clavi\'e and Atoof Shakir and Jonah Turner and Sean Lee and Aamir Shakir and Makoto P. Kato},
  eprint = {2602.11941},
  primaryclass = {cs.IR},
  title = {IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval},
  url = {https://arxiv.org/abs/2602.11941},
  year = {2026},
}
""",
        prompt={"query": "Retrieve music that matches the user's search query."},
    )


class IncompeBenchLenientRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IncompeBenchLenientRetrieval",
        description=(
            "IncompeBench-Lenient is a fine-grained text-to-music retrieval "
            "benchmark with 500 diverse search queries, 1,574 permissively "
            "licensed music snippets, and graded relevance judgements. The "
            "two variants share the same queries and corpus; unlike Strict, "
            "Lenient retains all positive relevance scores 1-3, including "
            "score-1 tangential matches."
        ),
        reference="https://arxiv.org/abs/2602.11941",
        dataset={
            "path": "mteb/IncompeBenchLenientRetrieval",
            "revision": "529032a89c6f8bd166f623322a736d528cf89ca7",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-29", "2026-02-12"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{clavie2026incompebench,
  archiveprefix = {arXiv},
  author = {Benjamin Clavi\'e and Atoof Shakir and Jonah Turner and Sean Lee and Aamir Shakir and Makoto P. Kato},
  eprint = {2602.11941},
  primaryclass = {cs.IR},
  title = {IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval},
  url = {https://arxiv.org/abs/2602.11941},
  year = {2026},
}
""",
        prompt={"query": "Retrieve music that matches the user's search query."},
    )
