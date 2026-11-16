from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_REPO = "MTEB-BR/stackoverflow-clustering"
_REVISION = "baea103af775d126bedd7e51e350c6e193557d0e"


class StackoverflowPtClustering(AbsTaskClustering):
    """Cluster pt.stackoverflow question titles into 10 technology tags."""

    metadata = TaskMetadata(
        name="StackoverflowPtClustering",
        description=(
            "Cluster native Brazilian-Portuguese technical question titles from "
            "the Portuguese Stack Overflow (pt.stackoverflow.com) into 10 "
            "technology tags (python, java, php, javascript, android, mysql, c#, "
            "html, css, c). Programming domain."
            " Task constructed for MTEB-BR (Stekel, 2026) from questions retrieved via the Stack Exchange API."
        ),
        reference="https://pt.stackoverflow.com/",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["Programming", "Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{stekel2026mtebbr,
  archiveprefix = {arXiv},
  author = {Stekel, Tardelli Ronan Coelho},
  eprint = {2607.04581},
  primaryclass = {cs.CL},
  title = {{MTEB-BR}: A Text Embedding Benchmark for {B}razilian {P}ortuguese},
  url = {https://arxiv.org/abs/2607.04581},
  year = {2026},
}

@misc{ptstackoverflow,
  author = {{Stack Exchange Inc.}},
  howpublished = {Community Q\&A site; data via the Stack Exchange API, content licensed CC-BY-SA-4.0},
  title = {Portuguese Stack Overflow (pt.stackoverflow.com)},
  url = {https://pt.stackoverflow.com/},
}
""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
