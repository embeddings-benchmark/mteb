from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class UrbanSound8kClustering(AbsTaskClustering):
    label_column_name: str = "classID"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="UrbanSound8kClustering",
        description=(
            "Audio clustering of UrbanSound8K: 8,732 urban sound recordings across "
            "10 environmental categories (air conditioner, car horn, children playing, "
            "dog bark, drilling, engine idling, gun shot, jackhammer, siren, street "
            "music). Evaluates unsupervised grouping of real-world urban soundscapes."
        ),
        reference="https://huggingface.co/datasets/danavery/urbansound8K",
        dataset={
            "path": "mteb/urbansound8K",
            "revision": "5b3867ddd7583a24871acdf6eb5494696ddd4cbc",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Salamon:UrbanSound:ACMMM:14,
  author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle = {Proceedings of the 22nd ACM international conference on Multimedia},
  organization = {ACM},
  pages = {1041--1044},
  title = {A Dataset and Taxonomy for Urban Sound Research},
  year = {2014},
}
""",
    )
