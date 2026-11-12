from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ImageCoDeT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ImageCoDeT2IRetrieval",
        description="Retrieve a specific video frame based on a precise caption.",
        reference="https://aclanthology.org/2022.acl-long.241.pdf",
        dataset={
            "path": "mteb/imagecode",
            "revision": "3d13625935ec7e207e7e870e33a033cf51d60626",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_3",
        date=("2022-05-22", "2022-05-27"),  # conference dates
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{krojer2022image,
  address = {Dublin, Ireland},
  author = {Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
  booktitle = {Proceedings of the 60th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2022.acl-long.241},
  editor = {Muresan, Smaranda and Nakov, Preslav and Villavicencio, Aline},
  month = {may},
  pages = {3426--3440},
  publisher = {Association for Computational Linguistics},
  title = {Image Retrieval from Contextual Descriptions},
  url = {https://aclanthology.org/2022.acl-long.241/},
  year = {2022},
}
""",
    )
