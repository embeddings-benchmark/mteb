from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.tasks.Retrieval.multilingual.NeuCLIR2023Retrieval import load_neuclir_data

_LANGUAGES = {
    "ru": ["rus-Cyrl"],
    "en": ["eng-Latn"],
}


class RuSciBenchCiteRetrieval(AbsTaskRetrieval, MultilingualTask):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCiteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cite_retrieval",
            "revision": "6cb447d02f41b8b775d5d9df7faf472f44d2f1db",
        },
        description="Retrieval of related scientific papers based on their title and abstract",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
        bibtex_citation="""
@article{vatolin2024ruscibench,
  author  = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  title   = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  journal = {Doklady Mathematics},
  year    = {2024},
  volume  = {110},
  number  = {1},
  pages   = {S251--S260},
  month   = {12},
  doi     = {10.1134/S1064562424602191},
  url     = {https://doi.org/10.1134/S1064562424602191},
  issn    = {1531-8362}
}""",
        prompt={
            "query": "Given a title and abstract of a scientific paper, retrieve the titles and abstracts of other relevant papers",
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_neuclir_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class RuSciBenchCociteRetrieval(MultilingualTask, AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciBenchCociteRetrieval",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_cocite_retrieval",
            "revision": "a5da47a245275669d2b6ddf8f96c5338dd2428b4",
        },
        description="Retrieval of related scientific papers based on their title and abstract",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
        bibtex_citation="""
@article{vatolin2024ruscibench,
  author  = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  title   = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  journal = {Doklady Mathematics},
  year    = {2024},
  volume  = {110},
  number  = {1},
  pages   = {S251--S260},
  month   = {12},
  doi     = {10.1134/S1064562424602191},
  url     = {https://doi.org/10.1134/S1064562424602191},
  issn    = {1531-8362}
}""",
        prompt={
            "query": "Given a title and abstract of a scientific paper, retrieve the titles and abstracts of other relevant papers",
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_neuclir_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
