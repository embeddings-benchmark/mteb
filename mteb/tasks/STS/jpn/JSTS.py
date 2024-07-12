from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class JSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="JSTS",
        dataset={
            "path": "shunk031/JGLUE",
            "revision": "50e79c314a7603ebc92236b66a0973d51a00ed8c",
            "name": "JSTS",
            "trust_remote_code": True,
        },
        description="Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset"
        "(Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.",
        reference="https://aclanthology.org/2022.lrec-1.317.pdf#page=2.00",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["jpn-Jpan"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2022-12-31"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}""",
        descriptive_stats={
            "n_samples": {"valudtion": 1457},
            "avg_character_length": {"valudtion": 46.34},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.rename_column("label", "score")
