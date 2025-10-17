from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class StanfordCarsZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="StanfordCarsZeroShot",
        description="Classifying car images from 96 makes.",
        reference="https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content",
        dataset={
            "path": "isaacchung/StanfordCars",
            "revision": "09ffe9bc7864d3f1e851529e5c4b7e05601a04fb",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2013-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Krause2013CollectingAL,
  author = {Jonathan Krause and Jia Deng and Michael Stark and Li Fei-Fei},
  title = {Collecting a Large-scale Dataset of Fine-grained Cars},
  url = {https://api.semanticscholar.org/CorpusID:16632981},
  year = {2013},
}
""",
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of a {name}."
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
