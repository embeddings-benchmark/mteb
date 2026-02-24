from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SOPI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SOPI2IRetrieval",
        description="Retrieve product photos of 22634 online products.",
        reference="https://paperswithcode.com/dataset/stanford-online-products",
        dataset={
            "path": "mteb/stanford-online-products",
            "revision": "9556adace66a20ac7ef034b6b9dd854c0ed9779f",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2019-07-17", "2019-07-17"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{oh2016deep,
  author = {Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {4004--4012},
  title = {Deep metric learning via lifted structured feature embedding},
  year = {2016},
}
""",
    )
    skip_first_result = True
