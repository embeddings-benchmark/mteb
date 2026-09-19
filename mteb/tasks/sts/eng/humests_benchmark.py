from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class HUMESTSBenchmark(AbsTaskSTS):
    metadata = TaskMetadata(
        name="HUMESTSBenchmark",
        dataset={
            "path": "mteb/mteb-human-stsbenchmark-sts",
            "revision": "cb05d5409f802e68d6ed39615ed67f7dc2235ac5",
        },
        description="Human evaluation subset of Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2021-01-01", "2021-12-31"),  # publication year
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{cer-etal-2017-semeval,
  address = {Vancouver, Canada},
  author = {Cer, Daniel  and
Diab, Mona  and
Agirre, Eneko  and
Lopez-Gazpio, I{\~n}igo  and
Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)},
  doi = {10.18653/v1/S17-2001},
  editor = {Bethard, Steven  and
Carpuat, Marine  and
Apidianaki, Marianna  and
Mohammad, Saif M.  and
Cer, Daniel  and
Jurgens, David},
  month = aug,
  pages = {1--14},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
  url = {https://aclanthology.org/S17-2001},
  year = {2017},
}
""",
        adapted_from=["STSBenchmark"],
    )

    min_score = 0
    max_score = 5
