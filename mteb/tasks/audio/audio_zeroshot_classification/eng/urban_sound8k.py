from mteb.abstasks.audio.abs_task_zero_shot_classification import (
    AbsTaskAudioZeroshotClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class UrbanSound8kZeroshotClassification(AbsTaskAudioZeroshotClassification):
    metadata = TaskMetadata(
        name="UrbanSound8kZeroshot",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/danavery/urbansound8K",
        dataset={
            "path": "danavery/urbansound8K",
            "revision": "8aa9177a0c5a6949ee4ee4b7fcabb01dfd4ae466",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{Salamon2014ADA,
  author = {Justin Salamon and Christopher Jacoby and Juan Pablo Bello},
  journal = {Proceedings of the 22nd ACM international conference on Multimedia},
  title = {A Dataset and Taxonomy for Urban Sound Research},
  url = {https://api.semanticscholar.org/CorpusID:207217115},
  year = {2014},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "classID"

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        return [
            "This is a sound of air conditioner",
            "This is a sound of car horn",
            "This is a sound of children playing",
            "This is a sound of dog bark",
            "This is a sound of drilling",
            "This is a sound of engine idling",
            "This is a sound of gun shot",
            "This is a sound of jackhammer",
            "This is a sound of siren",
            "This is a sound of street music",
        ]

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], label=self.label_column_name
        )
