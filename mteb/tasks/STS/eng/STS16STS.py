from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS16STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS16",
        dataset={
            "path": "mteb/sts16-sts",
            "revision": "4d8694f8f0e0100860b497b999b3dbed754a0513",
        },
        description="SemEval STS 2016 dataset",
        reference="https://www.aclweb.org/anthology/S16-1001",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{nakov-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 4: Sentiment Analysis in {T}witter",
    author = "Nakov, Preslav  and
      Ritter, Alan  and
      Rosenthal, Sara  and
      Sebastiani, Fabrizio  and
      Stoyanov, Veselin",
    editor = "Bethard, Steven  and
      Carpuat, Marine  and
      Cer, Daniel  and
      Jurgens, David  and
      Nakov, Preslav  and
      Zesch, Torsten",
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S16-1001",
    doi = "10.18653/v1/S16-1001",
    pages = "1--18",
}""",
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
