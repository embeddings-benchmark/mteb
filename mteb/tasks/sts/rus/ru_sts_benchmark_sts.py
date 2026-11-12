from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class RuSTSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="RuSTSBenchmarkSTS",
        dataset={
            "path": "ai-forever/ru-stsbenchmark-sts",
            "revision": "7cf24f325c6da6195df55bef3d86b5e0616f3018",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. "
        "The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["rus-Cyrl"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2018-01-01"),
        domains=["News", "Social", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
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
    )

    min_score = 0
    max_score = 5
