from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


class RuSciBenchBitextMining(AbsTaskBitextMining):
    fast_loading = True
    metadata = TaskMetadata(
        name="RuSciBenchBitextMining",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_bitext_mining",
            "revision": "e5840033c5cf2573932db027ac8001fe0a7eb6fa",
        },
        description="This task focuses on finding translations of scientific articles. The dataset is sourced from eLibrary, Russia's largest electronic library of scientific publications. Russian authors often provide English translations for their abstracts and titles, and the data consists of these paired titles and abstracts. The task evaluates a model's ability to match an article's Russian title and abstract to its English counterpart, or vice versa.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="BitextMining",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "ru-en": ["rus-Cyrl", "eng-Latn"],
            "en-ru": ["eng-Latn", "rus-Cyrl"],
        },
        main_score="f1",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Given the following title and abstract of the scientific article, find its translation",
    )
