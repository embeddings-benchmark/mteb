from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS16STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS16",
        dataset={
            "path": "mteb/sts16-sts",
            "revision": "4d8694f8f0e0100860b497b999b3dbed754a0513",
        },
        description="SemEval-2016 Task 4",
        reference="https://www.aclweb.org/anthology/S16-1001",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2015-10-01", "2015-12-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{nakov-etal-2016-semeval,
  address = {San Diego, California},
  author = {Nakov, Preslav  and
Ritter, Alan  and
Rosenthal, Sara  and
Sebastiani, Fabrizio  and
Stoyanov, Veselin},
  booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)},
  doi = {10.18653/v1/S16-1001},
  editor = {Bethard, Steven  and
Carpuat, Marine  and
Cer, Daniel  and
Jurgens, David  and
Nakov, Preslav  and
Zesch, Torsten},
  month = jun,
  pages = {1--18},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2016 Task 4: Sentiment Analysis in {T}witter},
  url = {https://aclanthology.org/S16-1001},
  year = {2016},
}
""",
    )

    min_score = 0
    max_score = 5
