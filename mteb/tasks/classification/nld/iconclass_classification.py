from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class IconclassClassification(AbsTaskClassification):
    samples_per_label = 32
    metadata = TaskMetadata(
        name="IconclassClassification",
        description="Iconclass is an iconographic thesaurus, which is widely used in the digital heritage domain to "
        "describe subjects depicted in artworks. The task is to classify the first layer of Iconclass",
        reference="https://dl.acm.org/doi/pdf/10.1145/3575865",
        dataset={
            "path": "clips/mteb-nl-iconclass-cls",
            "revision": "1cd02f1579dab39fedc95de8cc15fd620557a9f2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-01", "2020-05-01"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["Written", "Fiction"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{banar2023transfer,
  author = {Banar, Nikolay and Daelemans, Walter and Kestemont, Mike},
  journal = {ACM Journal on Computing and Cultural Heritage},
  number = {2},
  pages = {1--16},
  publisher = {ACM New York, NY},
  title = {Transfer learning for the visual arts: The multi-modal retrieval of iconclass codes},
  volume = {16},
  year = {2023},
}
""",
        prompt={
            "query": "Classificeer de gegeven titel van het kunstwerk in het juiste onderwerp of thema"
        },
    )
