from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

_REPO = "MTEB-BR/wikipedia-categories"
_REVISION = "ff433edc23c7e2dd3304da3fb7d0edba6356001a"


class WikipediaPTCategoriesClusteringP2P(AbsTaskClustering):
    """Cluster PT Wikipedia article paragraphs into 15 broad subject categories."""

    metadata = TaskMetadata(
        name="WikipediaPTCategoriesClusteringP2P",
        description=(
            "Cluster the first paragraph of Brazilian Portuguese Wikipedia "
            "articles into 15 broad subject categories (História, Geografia, "
            "Política, Esporte, Música, Cinema, Literatura, Religião, Ciência, "
            "Tecnologia, Animais, Plantas, Medicina, Filosofia, Astronomia). "
            "Articles sampled via MediaWiki API category traversal (depth 2) "
            "of curated root categories; first paragraph from the `extracts` API."
            " Task constructed for MTEB-BR (Stekel, 2026)."
        ),
        reference="https://huggingface.co/datasets/MTEB-BR/wikipedia-categories",
        dataset={
            "path": _REPO,
            "revision": _REVISION,
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
        date=("2026-05-24", "2026-05-24"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=["pt-BR"],
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
""",
    )

    input_column_name = "sentences"
    label_column_name = "label"
    # the datasets are small (<= ~5k documents): embed all of them instead of the default 4% sample
    max_fraction_of_documents_to_embed = None

    # mteb new AbsTaskClustering takes the dataset row-level as-is:
    # one (sentences: str, label: str) per row. Our dataset is already in
    # that shape from the build script, so no dataset_transform needed.
