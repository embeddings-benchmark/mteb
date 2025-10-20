from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class RonSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="RonSTS",
        dataset={
            "path": "mteb/RonSTS",
            "revision": "0e9021b8beb24b874230b4fc4e3d1612e72795e0",
        },
        description="High-quality Romanian translation of STSBenchmark.",
        reference="https://openreview.net/forum?id=JH61CD7afTv",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2021-01-31"),
        domains=["News", "Social", "Web", "Written"],  # web for image captions
        task_subtypes=None,
        license="cc-by-4.0",  # not specified
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{dumitrescu2021liro,
  author = {Dumitrescu, Stefan Daniel and Rebeja, Petru and Lorincz, Beata and Gaman, Mihaela and Avram, Andrei and Ilie, Mihai and Pruteanu, Andrei and Stan, Adriana and Rosia, Lorena and Iacobescu, Cristina and others},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 5
