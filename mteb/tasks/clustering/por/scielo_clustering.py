from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_REPO = "MTEB-BR/scielo-clustering"
_REVISION = "c84cafc4789b384b8cdd1d6bc2c22966e7e36f03"


class SciELOClusteringP2P(AbsTaskClustering):
    """Cluster Brazilian Portuguese scientific abstracts into 8 broad research areas."""

    metadata = TaskMetadata(
        name="SciELOClusteringP2P",
        description=(
            "Cluster Brazilian Portuguese scientific abstracts from the SciELO "
            "Brazil open-access library into 8 broad research areas (Health "
            "Sciences, Social Sciences, Agricultural Sciences, Biological/Life "
            "Sciences, Humanities & Arts, Engineering & Technology, Physical "
            "Sciences & Chemistry, Mathematics & Computer Science). Areas are "
            "consolidated from the Web-of-Science subject categories of each "
            "article; only pure CC-BY-4.0 articles are included."
            " Task constructed for MTEB-BR (Stekel, 2026) from the eduagarcia/scielo_abstracts corpus."
        ),
        reference="https://huggingface.co/datasets/MTEB-BR/scielo-clustering",
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
        date=("2026-06-24", "2026-06-24"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["brazilian"],
        sample_creation="found",
        adapted_from=None,
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

@misc{scielo_abstracts,
  author = {{SciELO}},
  howpublished = {HuggingFace dataset \texttt{eduagarcia/scielo\_abstracts}, derived from the SciELO open-access scientific library},
  title = {scielo\_abstracts (Hugging Face dataset)},
  url = {https://huggingface.co/datasets/eduagarcia/scielo_abstracts},
}
""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
    # the datasets are small (<= ~5k documents): embed all of them instead of the default 4% sample
    max_fraction_of_documents_to_embed = None
