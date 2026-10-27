from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetArxivQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VisRAGRetArxivQA",
        description="evaluate vision-based retrieval and generation on scientific figures and their surrounding context to preserve complex layouts and mathematical notations.",
        reference="https://aclanthology.org/2024.acl-long.775/",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetArxivQA",
            "revision": "b0da0f3f9677461eb78e34a7164596ff4f86bd52",
        },
        date=("2000-01-01", "2024-12-31"),
        domains=["Academic", "Non-fiction"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{li2024multimodalarxivdatasetimproving,
  address = {Bangkok, Thailand},
  author = {Li, Lei and Wang, Yuqi and Xu, Runxin and Wang, Peiyi and Feng, Xiachong and Kong, Lingpeng and Liu, Qi},
  booktitle = {Proceedings of the 62nd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2024.acl-long.775},
  editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  month = {aug},
  pages = {14369--14387},
  publisher = {Association for Computational Linguistics},
  title = {Multimodal {A}r{X}iv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models},
  url = {https://aclanthology.org/2024.acl-long.775/},
  year = {2024},
}""",
    )
