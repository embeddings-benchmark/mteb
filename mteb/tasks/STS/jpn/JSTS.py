from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class JSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="JSTS",
        dataset={
            "path": "mteb/JSTS",
            "revision": "5bac629e25799df4c9c80a6a5db983d6cba9e77d",
        },
        description="Japanese Semantic Textual Similarity Benchmark dataset construct from YJ Image Captions Dataset "
        + "(Miyazaki and Shimizu, 2016) and annotated by crowdsource annotators.",
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
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kurihara-etal-2022-jglue,
  abstract = {To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.},
  address = {Marseille, France},
  author = {Kurihara, Kentaro  and
Kawahara, Daisuke  and
Shibata, Tomohide},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
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
Piperidis, Stelios},
  month = jun,
  pages = {2957--2966},
  publisher = {European Language Resources Association},
  title = {{JGLUE}: {J}apanese General Language Understanding Evaluation},
  url = {https://aclanthology.org/2022.lrec-1.317},
  year = {2022},
}
""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
