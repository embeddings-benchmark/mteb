from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class StanfordCarsI2I(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StanfordCarsI2IRetrieval",
        description="Retrieve car images from 196 makes.",
        reference="https://pure.mpg.de/rest/items/item_2029263/component/file_2029262/content",
        dataset={
            "path": "isaacchung/stanford_cars_retrieval",
            "revision": "b27a0612211af3598bd11fe28af20928f20cce06",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2012-01-01", "2013-04-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
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
    skip_first_result = True
